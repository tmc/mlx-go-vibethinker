package mgpo

import (
	"testing"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// Phase-C property (DESIGN_RL_UPGRADE.md §2 Tier 3, FRPO): with BetaFuture = 0
// the future-KL term vanishes and LossFRPO is bit-identical to the substrate
// rl.GRPOLoss with the same config — the off-path is the unmodified baseline
// loss, even though LossFRPO is a sibling function. The control half (a nonzero
// BetaFuture) must change the loss, proving the knob is live.
func TestFRPOZeroBetaBitIdenticalToGRPOLoss(t *testing.T) {
	current, old, ref, mask := toyRollouts(t)
	adv, err := AdvantageTensor([]float64{0.5, -0.5, 1.0, -1.0})
	if err != nil {
		t.Fatalf("AdvantageTensor: %v", err)
	}

	for _, drGRPO := range []bool{true, false} {
		cfg := rl.DefaultGRPOConfig()
		cfg.DrGRPO = drGRPO

		want, err := rl.GRPOLoss(current, old, ref, adv, mask, cfg)
		if err != nil {
			t.Fatalf("GRPOLoss: %v", err)
		}
		got, err := LossFRPO(current, old, ref, adv, mask, cfg, FRPOConfig{}) // BetaFuture 0
		if err != nil {
			t.Fatalf("LossFRPO beta=0: %v", err)
		}
		if err := mlx.Eval(want, got); err != nil {
			t.Fatalf("eval: %v", err)
		}
		if a, b := mlx.ArrayItemFloat32(want), mlx.ArrayItemFloat32(got); a != b {
			t.Fatalf("DrGRPO=%v: LossFRPO beta=0 %v != GRPOLoss %v (must be bit-identical)", drGRPO, b, a)
		}

		// Control: a nonzero future-KL weight must move the loss.
		on, err := LossFRPO(current, old, ref, adv, mask, cfg, FRPOConfig{BetaFuture: 0.1})
		if err != nil {
			t.Fatalf("LossFRPO beta>0: %v", err)
		}
		if err := mlx.Eval(on); err != nil {
			t.Fatalf("eval on: %v", err)
		}
		if a, b := mlx.ArrayItemFloat32(want), mlx.ArrayItemFloat32(on); a == b {
			t.Fatalf("DrGRPO=%v: LossFRPO beta=0.1 %v matched baseline %v; future-KL knob is inert", drGRPO, b, a)
		}
	}
}

// Phase-C cross-cutting invariant: LossFRPOScaled preserves the MGPO no-op rule.
// With a zero FRPOConfig it is bit-identical to LossOpt (w_ME multiplies the
// advantage, never the reward); with a nonzero BetaFuture it adds the future-KL
// term on top of that same scaled advantage.
func TestFRPOScaledNoOpRulePreserved(t *testing.T) {
	current, old, ref, mask := toyRollouts(t)
	rewards := [][]float64{{1, 0, 1, 0}}
	const lambda = 0.7
	cfg := rl.DefaultGRPOConfig()

	for _, opts := range []Options{{}, {DrGRPOAdvantage: true}} {
		want, err := LossOpt(current, old, ref, mask, rewards, lambda, cfg, opts)
		if err != nil {
			t.Fatalf("LossOpt: %v", err)
		}
		got, err := LossFRPOScaled(current, old, ref, mask, rewards, lambda, cfg, opts, FRPOConfig{})
		if err != nil {
			t.Fatalf("LossFRPOScaled beta=0: %v", err)
		}
		if err := mlx.Eval(want, got); err != nil {
			t.Fatalf("eval: %v", err)
		}
		if a, b := mlx.ArrayItemFloat32(want), mlx.ArrayItemFloat32(got); a != b {
			t.Fatalf("opts=%+v: LossFRPOScaled beta=0 %v != LossOpt %v (no-op rule broken)", opts, b, a)
		}
	}
}

// Phase-C property: the future-KL term respects the mask. A token masked out
// (mask=0) contributes nothing to the reverse-cumsum return-to-go, so masking
// the whole sequence makes the future-KL term vanish and LossFRPO collapses to
// the baseline regardless of BetaFuture. (A fully-masked sequence has zero loss
// in the substrate too; the point is that beta does not leak through the mask.)
func TestFRPOFutureTermRespectsMask(t *testing.T) {
	current, old, ref, _ := toyRollouts(t)
	adv, err := AdvantageTensor([]float64{0.5, -0.5, 1.0, -1.0})
	if err != nil {
		t.Fatalf("AdvantageTensor: %v", err)
	}
	// Zero mask: no generated tokens.
	zeroMask := mlx.NewArray(make([]float32, 4*3), 4, 3)
	cfg := rl.DefaultGRPOConfig()
	cfg.DrGRPO = true // avoid divide-by-zero token count in the length-norm branch

	base, err := LossFRPO(current, old, ref, adv, zeroMask, cfg, FRPOConfig{})
	if err != nil {
		t.Fatalf("LossFRPO beta=0 zero-mask: %v", err)
	}
	withBeta, err := LossFRPO(current, old, ref, adv, zeroMask, cfg, FRPOConfig{BetaFuture: 5.0})
	if err != nil {
		t.Fatalf("LossFRPO beta>0 zero-mask: %v", err)
	}
	if err := mlx.Eval(base, withBeta); err != nil {
		t.Fatalf("eval: %v", err)
	}
	if a, b := mlx.ArrayItemFloat32(base), mlx.ArrayItemFloat32(withBeta); a != b {
		t.Fatalf("future-KL leaked through a zero mask: beta=5 %v != beta=0 %v", b, a)
	}
}
