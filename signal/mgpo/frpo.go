package mgpo

import (
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// FRPO Future-KL regularization (arXiv 2601.10201). GRPO's per-token KL penalty
// is local: it regularizes each token's divergence from the reference in
// isolation and ignores the divergence that accumulates over the rest of the
// sequence. FRPO adds a causal "return-to-go" of the per-token log-ratio — the
// reverse cumulative sum, from each token to the end of the sequence — so the
// objective accounts for future policy drift, not just the current token's.
//
// The substrate rl.GRPOLoss computes the per-token log-ratio internally and
// exposes no injection point for an extra per-token term (its advantages
// argument is per-sequence, broadcast across tokens). LossFRPO is therefore a
// sibling loss that reproduces the substrate surrogate exactly and adds the
// future-KL term on top, rather than editing the substrate. With BetaFuture = 0
// the added term vanishes and LossFRPO is bit-identical to rl.GRPOLoss with the
// same config — the off-path is unchanged.

// FRPOConfig configures the FRPO future-KL term layered onto the GRPO surrogate.
// Its zero value (BetaFuture = 0) adds nothing, so [LossFRPO] with a zero
// FRPOConfig equals the baseline GRPO loss.
type FRPOConfig struct {
	// BetaFuture weights the future-KL return-to-go term. 0 disables it
	// (baseline GRPO). The paper uses a small positive value; the term is
	// subtracted from the per-token objective like the local KL penalty, so a
	// larger BetaFuture more strongly discourages cumulative future drift.
	BetaFuture float64
}

// LossFRPO computes the GRPO clipped surrogate with KL penalty (identical to
// rl.GRPOLoss) plus the FRPO future-KL regularization term. The advantages here
// are the MGPO-scaled advantages: w_ME has already multiplied the group-relative
// advantage, so the no-op rule is preserved exactly as in [Loss] — FRPO adds a
// per-token objective term and never touches the advantage or the reward.
//
// All tensors match rl.GRPOLoss: current, old, ref are per-token log-probs (old
// and ref stop-gradiented), advantages are per-sequence broadcast across tokens,
// mask is 1 on generated tokens. The future-KL term for token t is
// frpo.BetaFuture times the reverse cumulative sum over t of the masked per-token
// log-ratio (current − old), so each token is penalized by the log-ratio mass
// from itself to the end of its sequence. With frpo.BetaFuture = 0 the term is
// zero and the result equals rl.GRPOLoss(current, old, ref, advantages, mask,
// config) bit-for-bit.
func LossFRPO(current, old, ref, advantages, mask *mlx.Array, config rl.GRPOConfig, frpo FRPOConfig) (*mlx.Array, error) {
	if frpo.BetaFuture == 0 {
		// Off-path: exactly the substrate surrogate, no future-KL term.
		return rl.GRPOLoss(current, old, ref, advantages, mask, config)
	}

	// --- Reproduce the substrate per-token surrogate (see mlx-go-rl/grpo.go). ---
	logRatio := mlx.Subtract(current, old)
	defer logRatio.Free()

	ratio := mlx.Exp(logRatio)
	defer ratio.Free()

	clipLowEps := config.ClipEpsLow
	if clipLowEps == 0 {
		clipLowEps = config.ClipEps
	}
	clipHighEps := config.ClipEpsHigh
	if clipHighEps == 0 {
		clipHighEps = config.ClipEps
	}
	clipLow := mlx.NewScalar(float32(1.0 - clipLowEps))
	defer clipLow.Free()
	clipHigh := mlx.NewScalar(float32(1.0 + clipHighEps))
	defer clipHigh.Free()

	clipped := mlx.Clip(ratio, clipLow, clipHigh)
	defer clipped.Free()

	ratioAdv := mlx.Multiply(ratio, advantages)
	defer ratioAdv.Free()
	clippedAdv := mlx.Multiply(clipped, advantages)
	defer clippedAdv.Free()

	surr := mlx.Minimum(ratioAdv, clippedAdv)
	defer surr.Free()

	// Local KL penalty: kl = exp(d) - d - 1, d = clamp(ref - current, -C, C).
	diff := mlx.Subtract(ref, current)
	defer diff.Free()
	klClampLow := mlx.NewScalar(float32(-config.KLClamp))
	defer klClampLow.Free()
	klClampHigh := mlx.NewScalar(float32(config.KLClamp))
	defer klClampHigh.Free()
	d := mlx.Clip(diff, klClampLow, klClampHigh)
	defer d.Free()
	expD := mlx.Exp(d)
	defer expD.Free()
	expDMinusD := mlx.Subtract(expD, d)
	defer expDMinusD.Free()
	one := mlx.NewScalar(float32(1.0))
	defer one.Free()
	kl := mlx.Subtract(expDMinusD, one)
	defer kl.Free()
	klCoeffArr := mlx.NewScalar(float32(config.KLCoeff))
	defer klCoeffArr.Free()
	klScaled := mlx.Multiply(klCoeffArr, kl)
	defer klScaled.Free()

	perToken := mlx.Subtract(surr, klScaled)
	defer perToken.Free()

	// --- FRPO future-KL term: reverse cumulative sum of the masked log-ratio. ---
	// Mask the log-ratio so prompt/pad tokens contribute nothing to the
	// return-to-go, then take the reverse (suffix) cumulative sum along the
	// token axis: futureLogRatio[i,t] = Σ_{s≥t} mask[i,s]·logRatio[i,s].
	maskedLogRatio := mlx.Multiply(logRatio, mask)
	defer maskedLogRatio.Free()
	futureLogRatio := mlx.Cumsum(maskedLogRatio, -1, true, true) // reverse, inclusive
	defer futureLogRatio.Free()

	betaArr := mlx.NewScalar(float32(frpo.BetaFuture))
	defer betaArr.Free()
	futureKL := mlx.Multiply(betaArr, futureLogRatio)
	defer futureKL.Free()

	// Subtract the future-KL term from the per-token objective, like the local
	// KL penalty: a positive cumulative future log-ratio (the policy growing
	// probability over the rest of the sequence) is discouraged.
	objective := mlx.Subtract(perToken, futureKL)
	defer objective.Free()

	// --- Reduce exactly as the substrate does. ---
	masked := mlx.Multiply(objective, mask)
	defer masked.Free()

	seqSum := mlx.SumAxis(masked, -1, false)
	defer seqSum.Free()

	seqLoss := seqSum
	if !config.DrGRPO {
		tokenCounts := mlx.SumAxis(mask, -1, false)
		defer tokenCounts.Free()
		seqLoss = mlx.Divide(seqSum, tokenCounts)
		defer seqLoss.Free()
	}

	seqMean := mlx.Mean(seqLoss, false)
	defer seqMean.Free()

	loss := mlx.Negative(seqMean)
	return loss, nil
}

// LossFRPOScaled is the FRPO counterpart of [Loss]: it scales the rewards'
// advantages by w_ME (via opts, so Dr.GRPO composes), materializes them, and
// calls [LossFRPO]. With a zero FRPOConfig it is bit-identical to [LossOpt]; with
// a nonzero BetaFuture it adds the future-KL term. The flattened advantage order
// must match the sequence order of the tensors (group-major), as in [Loss].
func LossFRPOScaled(current, old, ref, mask *mlx.Array, rewards [][]float64, lambda float64, config rl.GRPOConfig, opts Options, frpo FRPOConfig) (*mlx.Array, error) {
	scaled, err := ScaledAdvantagesOpt(rewards, lambda, opts)
	if err != nil {
		return nil, err
	}
	advArr, err := AdvantageTensor(FlattenAdvantages(scaled))
	if err != nil {
		return nil, err
	}
	loss, err := LossFRPO(current, old, ref, advArr, mask, opts.applyClip(config), frpo)
	if err != nil {
		return nil, fmt.Errorf("mgpo: LossFRPO: %w", err)
	}
	return loss, nil
}
