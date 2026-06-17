package mgpo

import rl "github.com/tmc/mlx-go/examples/mlx-go-rl"

// Options selects the Tier-1 post-GRPO refinements layered onto MGPO. Its zero
// value reproduces the DESIGN.md baseline exactly: a standard, std-normalized
// group advantage and symmetric PPO clipping. Every field is opt-in, so the
// baseline reproduction is never disturbed by default (DESIGN_RL_UPGRADE.md §4).
//
// The two refinements are independent and compose with w_ME and Long2Short:
//
//   - DrGRPOAdvantage removes the question-level std divisor from the advantage
//     (Dr.GRPO, arXiv 2503.20783): Aᵢ = rᵢ − mean(r) instead of
//     (rᵢ − mean)/std. This is the advantage half of Dr.GRPO. The loss half —
//     dropping the per-sequence 1/|oᵢ| length divisor — is a separate switch on
//     the rl.GRPOConfig passed to [Loss] (config.DrGRPO), already plumbed by the
//     substrate. The two halves are deliberately decoupled.
//
//   - ClipEpsLow and ClipEpsHigh decouple the PPO clip range (DAPO Clip-Higher,
//     arXiv 2503.14476): clip(r, 1−εlow, 1+εhigh). A zero field falls back to the
//     config's symmetric ClipEps (the substrate applies the same fallback), so
//     εlow = εhigh reproduces symmetric clipping bit-for-bit.
type Options struct {
	// DrGRPOAdvantage uses rl.GroupAdvantageDrGRPO (no std normalization) in
	// place of rl.GroupAdvantage. Default false (std-normalized, today's
	// baseline).
	DrGRPOAdvantage bool

	// ClipEpsLow and ClipEpsHigh set the asymmetric clip range applied by
	// rl.GRPOLoss. Zero means "use the config's symmetric ClipEps". Recommended
	// Clip-Higher values are 0.2 and 0.28.
	ClipEpsLow  float64
	ClipEpsHigh float64
}

// groupAdvantage returns the unweighted per-group advantages for these options:
// the no-std Dr.GRPO advantage when DrGRPOAdvantage is set, otherwise the
// std-normalized GRPO advantage. w_ME is applied by the caller, after this, so
// it always multiplies the advantage (never the raw reward) regardless of which
// normalization is chosen — preserving the MGPO no-op rule.
func (o Options) groupAdvantage(rewards [][]float64) [][]float64 {
	if o.DrGRPOAdvantage {
		return rl.GroupAdvantageDrGRPO(rewards)
	}
	return rl.GroupAdvantage(rewards)
}

// applyClip returns a copy of config with the Clip-Higher overrides applied. A
// zero override leaves the field zero, which rl.GRPOLoss resolves to the
// symmetric config.ClipEps — so the zero Options yields config unchanged and a
// clip identical to today.
func (o Options) applyClip(config rl.GRPOConfig) rl.GRPOConfig {
	if o.ClipEpsLow != 0 {
		config.ClipEpsLow = o.ClipEpsLow
	}
	if o.ClipEpsHigh != 0 {
		config.ClipEpsHigh = o.ClipEpsHigh
	}
	return config
}
