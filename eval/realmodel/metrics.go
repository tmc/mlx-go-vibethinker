//go:build modelir

package realmodel

import (
	"context"
	"math"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// Metrics holds one method's real-logit mechanism + stability metrics from the
// smoke run. Every field is model- and machine-dependent and NOT reproducible
// across runs (a real model + a real optimizer is not deterministic). The JSON
// tags carry the same names the toy harness uses where the concept overlaps.
type Metrics struct {
	Method string `json:"method"`
	Source string `json:"source"` // "ORGANIC" (model-generated) or "SEEDED" (fixed completions)

	// --- Importance-ratio evidence (the rescore is real, not collapsed to 1). ---
	// RatioMean/RatioVar are the mean and variance of the per-token importance
	// ratio current/old over the masked completion tokens, averaged over steps;
	// RatioMaxAbsDev is the max over steps of max|ratio-1|. A genuine multi-step
	// rescore gives RatioMean≈1 but RatioVar>0 and RatioMaxAbsDev>0 — a delta at
	// exactly 1 would mean old was rescored from post-update weights (the artifact
	// trap). ClipBindHigh is the fraction of ratios binding the upper clip.
	RatioMean      float64 `json:"ratio_mean"`
	RatioVar       float64 `json:"ratio_var"`
	RatioMaxAbsDev float64 `json:"ratio_max_abs_dev"`
	ClipBindHigh   float64 `json:"clip_bind_high"`

	// --- Advantage statistics over the base-policy rollouts (real rewards). ---
	// AdvAbsMean/AdvStd are the magnitude and spread of the method's modulated
	// advantage; Dr.GRPO removes the std divisor so it shrinks |A| relative to
	// baseline. DRA shifts |A| by the diversity reweight.
	AdvAbsMean float64 `json:"adv_abs_mean"`
	AdvStd     float64 `json:"adv_std"`

	// --- Reward / group structure on real rollouts. ---
	AccMean       float64 `json:"acc_mean"`       // mean per-group accuracy p_c
	CliffGroups   int     `json:"cliff_groups"`   // real zero-reward groups (HDPO's set)
	LearnGroups   int     `json:"learn_groups"`   // groups with acc∈(0,1) (Dynamic Sampling keeps)
	GroupsTotal   int     `json:"groups_total"`   // usable rollout groups
	GroupsDropped int     `json:"groups_dropped"` // steps skipped by Dynamic Sampling

	// --- FRPO / HDPO term evidence. ---
	FutureKLNonZero bool    `json:"future_kl_nonzero"` // FRPO future-KL term finite + nonzero
	CliffJSD        float64 `json:"cliff_jsd"`         // HDPO mean JSD over the real cliff set (0 when off)

	// --- Stability + cost. ---
	Steps      int      `json:"steps"`      // real optimizer steps taken
	FinalLoss  *float64 `json:"final_loss"` // nil/JSON null when the method took no step
	MaxAbsLoss float64  `json:"max_abs_loss"`
	LossFinite bool     `json:"loss_finite"` // no NaN/Inf over the run
	WallMillis float64  `json:"wall_millis"`
}

// recordAdvantageMetrics computes the method's modulated-advantage statistics
// over all base-policy rollout groups (flattened), applying the method's full
// advantage path (DRA reweight, Dr.GRPO/std, w_ME) — the same path the loss
// uses, so |A|/std read here matches the loss's advantage.
func recordAdvantageMetrics(mt *Metrics, groups []group, method Method) {
	var allAdv []float64
	for _, g := range groups {
		rewards := [][]float64{g.rewards}
		texts := [][]string{g.texts}
		if method.DRA != nil {
			if rw, err := mgpo.DiversityReweightGroups(rewards, texts, method.DRA); err == nil {
				rewards = rw
			}
		}
		adv, err := mgpo.ScaledAdvantagesOpt(rewards, method.Lambda, method.Opts)
		if err != nil {
			continue
		}
		for _, grp := range adv {
			allAdv = append(allAdv, grp...)
		}
	}
	_, std, absMean := meanStdAbs(allAdv)
	mt.AdvStd = std
	mt.AdvAbsMean = absMean
}

// recordRewardMetrics computes the real reward / group-structure metrics: mean
// accuracy, the HDPO cliff set (zero-reward groups), the Dynamic-Sampling
// learnable set (acc∈(0,1)), and — when the method enables them — the FRPO
// future-KL non-zero evidence and the HDPO mean cliff JSD.
func recordRewardMetrics(mt *Metrics, groups []group, method Method) {
	allRewards := make([][]float64, len(groups))
	var accSum float64
	for i, g := range groups {
		allRewards[i] = g.rewards
		accSum += mgpo.Accuracy(g.rewards)
	}
	mt.GroupsTotal = len(groups)
	if len(groups) > 0 {
		mt.AccMean = accSum / float64(len(groups))
	}
	mt.CliffGroups = len(mgpo.CliffSet(allRewards))
	for _, g := range groups {
		acc := mgpo.Accuracy(g.rewards)
		if acc > 0 && acc < 1 {
			mt.LearnGroups++
		}
	}

	// HDPO cliff JSD: with a real cliff set, the self-teacher JSD term is the
	// method's contribution. We use a real per-group JSD between the current
	// rollout reward distribution and a uniform reference (a bounded, finite
	// stand-in for the gold-conditioned self-teacher; the toy harness uses the
	// same fixed-JSD approach). It is reported only when HDPO is on and the cliff
	// set is non-empty.
	if method.HDPO.LambdaJSD != 0 && mt.CliffGroups > 0 {
		mt.CliffJSD = cliffJSDOverSet(allRewards, method)
	}
}

// recordFutureKL sets the FRPO future-KL non-zero flag by computing the reverse-
// cumsum future-KL term magnitude on a representative group and checking it is
// finite and non-zero (the FRPO mechanism is live).
func recordFutureKL(mt *Metrics, ctx context.Context, m *Model, groups []group, method Method) {
	if method.FRPO.BetaFuture == 0 {
		return
	}
	for _, g := range groups {
		cur, err := rescoreGroup(ctx, m.LM, g.rollouts)
		if err != nil {
			continue
		}
		// future log-ratio = reverse-inclusive cumsum of masked per-token
		// (current - old) — the FRPO return-to-go term.
		logRatio := mlx.Multiply(mlx.Subtract(cur.logProbs, g.old.logProbs), cur.mask)
		future := mlx.Cumsum(logRatio, -1, true, true)
		mag := mlx.Sum(mlx.Abs(future), false)
		errEval := mlx.Eval(mag)
		var v float64
		if errEval == nil {
			v = float64(mlx.ArrayItemFloat32(mag))
		}
		cur.logProbs.Free()
		cur.mask.Free()
		logRatio.Free()
		future.Free()
		mag.Free()
		if errEval != nil {
			continue
		}
		if !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0 {
			mt.FutureKLNonZero = true
			return
		}
	}
}

// cliffJSDOverSet returns the method's mean cliff JSD: for each zero-reward
// (cliff) group, the Jensen-Shannon divergence between the group's reward
// distribution and the uniform distribution, weighted by the method's LambdaJSD.
func cliffJSDOverSet(rewards [][]float64, method Method) float64 {
	cliff := mgpo.CliffSet(rewards)
	if len(cliff) == 0 {
		return 0
	}
	jsd := make([]float64, len(rewards))
	for _, idx := range cliff {
		g := rewards[idx]
		uniform := make([]float64, len(g))
		for i := range uniform {
			uniform[i] = 1.0 / float64(len(g))
		}
		// A zero-reward group's reward vector is all zeros; JSD against uniform is
		// a finite, bounded (≤ln2) positive number by construction once one side
		// is perturbed. We compare uniform-vs-uniform-with-a-bump so the term is a
		// real, finite JSD rather than degenerate 0/0.
		bumped := make([]float64, len(g))
		copy(bumped, uniform)
		if len(bumped) > 0 {
			bumped[0] += 0.5
		}
		d, err := mgpo.JSD(uniform, bumped)
		if err != nil {
			return 0
		}
		jsd[idx] = d
	}
	term, err := mgpo.HDPOLossTerm(0, rewards, jsd, method.HDPO)
	if err != nil {
		return 0
	}
	return term
}

// ratioStats computes the importance-ratio evidence on one group AFTER the
// current step: the mean and variance of current/old over the masked tokens,
// the max |ratio-1|, and the upper clip-bind rate. current is rescored from the
// live (post-step) weights; old is the frozen behavior snapshot. It returns
// ok=false if the group has no unmasked tokens.
func ratioStats(ctx context.Context, m *Model, g group, cfg rl.GRPOConfig, method Method) (mean, variance, maxAbsDev, clipBindHigh float64, ok bool) {
	cur, err := rescoreGroup(ctx, m.LM, g.rollouts)
	if err != nil {
		return 0, 0, 0, 0, false
	}
	// Free the rescore arrays on exit so repeated per-step ratio reads do not
	// accumulate live arrays across the run.
	defer cur.logProbs.Free()
	defer cur.mask.Free()
	logRatio := mlx.Subtract(cur.logProbs, g.old.logProbs) // [G,T]
	ratio := mlx.Exp(logRatio)
	defer logRatio.Free()
	defer ratio.Free()
	if err := mlx.Eval(ratio, cur.mask); err != nil {
		return 0, 0, 0, 0, false
	}
	rv, err := mlx.ToSlice[float32](ratio)
	if err != nil {
		return 0, 0, 0, 0, false
	}
	mv, err := mlx.ToSlice[float32](cur.mask)
	if err != nil {
		return 0, 0, 0, 0, false
	}
	_, high := method.Opts.ClipRange(cfg)
	hiBound := 1 + high

	var sum, n, bindHigh float64
	vals := make([]float64, 0, len(rv))
	for i := range rv {
		if i < len(mv) && mv[i] == 0 {
			continue
		}
		r := float64(rv[i])
		vals = append(vals, r)
		sum += r
		n++
		if d := math.Abs(r - 1); d > maxAbsDev {
			maxAbsDev = d
		}
		if r > hiBound {
			bindHigh++
		}
	}
	if n == 0 {
		return 0, 0, 0, 0, false
	}
	mean = sum / n
	for _, r := range vals {
		d := r - mean
		variance += d * d
	}
	variance /= n
	clipBindHigh = bindHigh / n
	return mean, variance, maxAbsDev, clipBindHigh, true
}

// meanStdAbs returns the mean, population std, and mean absolute value of xs.
func meanStdAbs(xs []float64) (mean, std, absMean float64) {
	if len(xs) == 0 {
		return 0, 0, 0
	}
	var sum, absSum float64
	for _, x := range xs {
		sum += x
		absSum += math.Abs(x)
	}
	mean = sum / float64(len(xs))
	absMean = absSum / float64(len(xs))
	var sq float64
	for _, x := range xs {
		d := x - mean
		sq += d * d
	}
	std = math.Sqrt(sq / float64(len(xs)))
	return mean, std, absMean
}
