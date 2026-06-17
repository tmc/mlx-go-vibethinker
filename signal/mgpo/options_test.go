package mgpo

import (
	"math"
	"testing"

	"github.com/tmc/mlx-go-vibethinker/signal/long2short"
	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// Phase-A property (DESIGN_RL_UPGRADE.md §3a): with Dr.GRPO off (the zero
// Options) the modulated advantage is bit-identical to today's ScaledAdvantages,
// for every λ — the baseline reproduction is undisturbed. The control half
// confirms Dr.GRPO ON genuinely differs (so the knob is live, not inert).
func TestDrGRPOOffBitIdenticalToBaseline(t *testing.T) {
	rewards := [][]float64{
		{1, 0, 1, 1},
		{0, 0, 1, 0},
		{1, 1, 1, 1}, // degenerate group (std=0) — exercises the guard path
	}
	for _, lambda := range []float64{0, 0.5, 1.0, 4.0} {
		want, err := ScaledAdvantages(rewards, lambda)
		if err != nil {
			t.Fatalf("ScaledAdvantages: %v", err)
		}
		got, err := ScaledAdvantagesOpt(rewards, lambda, Options{}) // Dr.GRPO off
		if err != nil {
			t.Fatalf("ScaledAdvantagesOpt: %v", err)
		}
		for i := range want {
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("λ=%v group %d rollout %d: opt-off %v != baseline %v (must be bit-identical)",
						lambda, i, j, got[i][j], want[i][j])
				}
			}
		}
		// Control: Dr.GRPO ON must differ from the std-normalized baseline on a
		// group with nonzero, non-unit std (group 0). If it didn't, the knob
		// would be doing nothing.
		drg, err := ScaledAdvantagesOpt(rewards, lambda, Options{DrGRPOAdvantage: true})
		if err != nil {
			t.Fatalf("ScaledAdvantagesOpt DrGRPO: %v", err)
		}
		if drg[0][0] == want[0][0] {
			t.Fatalf("λ=%v: Dr.GRPO ON produced the same advantage %v as std-normalized baseline; knob is inert",
				lambda, drg[0][0])
		}
	}
}

// Phase-A property (DESIGN_RL_UPGRADE.md §3b): Clip-Higher with ε_low == ε_high
// is bit-identical to symmetric clipping, and an asymmetric ε_low != ε_high
// genuinely changes the loss (the knob is live).
func TestClipHigherSymmetricEqualsBaseline(t *testing.T) {
	current, old, ref, mask := toyRollouts(t)
	rewards := [][]float64{{1, 0, 1, 0}}

	// Baseline: symmetric clip via config.ClipEps, no option override.
	base := rl.DefaultGRPOConfig()
	base.ClipEps = 0.25
	base.ClipEpsLow = 0
	base.ClipEpsHigh = 0
	wantLoss, err := Loss(current, old, ref, mask, rewards, 0.7, base)
	if err != nil {
		t.Fatalf("Loss baseline: %v", err)
	}

	// Clip-Higher with ε_low == ε_high == 0.25 must reproduce it bit-for-bit.
	eqLoss, err := LossOpt(current, old, ref, mask, rewards, 0.7, base,
		Options{ClipEpsLow: 0.25, ClipEpsHigh: 0.25})
	if err != nil {
		t.Fatalf("LossOpt symmetric: %v", err)
	}
	if err := mlx.Eval(wantLoss, eqLoss); err != nil {
		t.Fatalf("eval: %v", err)
	}
	if a, b := mlx.ArrayItemFloat32(wantLoss), mlx.ArrayItemFloat32(eqLoss); a != b {
		t.Fatalf("ε_low==ε_high loss %v != symmetric baseline %v (must be bit-identical)", b, a)
	}

	// Control: the recommended asymmetric Clip-Higher (0.2 / 0.28) must differ.
	asym, err := LossOpt(current, old, ref, mask, rewards, 0.7, base,
		Options{ClipEpsLow: 0.2, ClipEpsHigh: 0.28})
	if err != nil {
		t.Fatalf("LossOpt asymmetric: %v", err)
	}
	if err := mlx.Eval(asym); err != nil {
		t.Fatalf("eval asym: %v", err)
	}
	if a, b := mlx.ArrayItemFloat32(wantLoss), mlx.ArrayItemFloat32(asym); a == b {
		t.Fatalf("asymmetric Clip-Higher loss %v matched symmetric %v; the clip knob is inert", b, a)
	}
}

// Phase-A property (DESIGN_RL_UPGRADE.md §3c): the MGPO no-op rule holds under
// Dr.GRPO too — w_ME multiplies the (un-std-normalized) advantage, never the raw
// reward. At λ=0 the result is exactly the plain Dr.GRPO advantage; at λ>0 it is
// exactly w_ME(p_c)·A_DrGRPO per group.
func TestNoOpRulePreservedUnderDrGRPO(t *testing.T) {
	rewards := [][]float64{
		{1, 0, 1, 1}, // p_c = 0.75
		{1, 0, 0, 0}, // p_c = 0.25
	}
	baseDrGRPO := rl.GroupAdvantageDrGRPO(rewards)

	// λ=0: w_ME ≡ 1, so the modulated advantage is exactly the Dr.GRPO advantage
	// — proving the weight multiplies the advantage (here by 1), not the reward.
	at0, err := ScaledAdvantagesOpt(rewards, 0, Options{DrGRPOAdvantage: true})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt λ=0: %v", err)
	}
	for i := range baseDrGRPO {
		for j := range baseDrGRPO[i] {
			if at0[i][j] != baseDrGRPO[i][j] {
				t.Fatalf("λ=0 Dr.GRPO group %d rollout %d: %v != plain Dr.GRPO advantage %v",
					i, j, at0[i][j], baseDrGRPO[i][j])
			}
		}
	}

	// λ>0: each group's advantage is exactly w_ME(p_c)·A_DrGRPO. We reconstruct
	// the expected value from the public Weight and compare — if the code scaled
	// the reward instead, the no-std advantage would not factor this way.
	const lambda = 3.0
	at, err := ScaledAdvantagesOpt(rewards, lambda, Options{DrGRPOAdvantage: true})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt λ>0: %v", err)
	}
	for i := range rewards {
		w, err := Weight(lambda, Accuracy(rewards[i]))
		if err != nil {
			t.Fatalf("Weight: %v", err)
		}
		for j := range rewards[i] {
			want := w * baseDrGRPO[i][j]
			if math.Abs(at[i][j]-want) > 1e-12 {
				t.Fatalf("group %d rollout %d: %v != w_ME·A_DrGRPO %v", i, j, at[i][j], want)
			}
		}
	}
}

// Phase-A property (DESIGN_RL_UPGRADE.md §3d): Long2Short's brevity reshaping is
// zero-sum over the correct set C, and that property is independent of the mgpo
// advantage normalization (std vs Dr.GRPO no-std) applied downstream. The
// std-removal must not silently rescale the Long2Short shift.
func TestLong2ShortZeroSumUnderNormalizationChange(t *testing.T) {
	traces := []long2short.Trace{
		{Reward: 1, Length: 100, Correct: true},
		{Reward: 1, Length: 200, Correct: true},
		{Reward: 1, Length: 400, Correct: true},
		{Reward: 0, Length: 150, Correct: false}, // incorrect: untouched
	}
	original := make([]float64, len(traces))
	for i, tr := range traces {
		original[i] = tr.Reward
	}
	reshaped, err := long2short.Reshape(traces, long2short.DefaultLambda)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}

	// Δ over the correct set C must sum to zero (the brevity shift is zero-sum).
	var deltaC float64
	for i, tr := range traces {
		if tr.Correct {
			deltaC += reshaped[i] - original[i]
		} else if reshaped[i] != original[i] {
			t.Fatalf("incorrect trace %d reward changed: %v -> %v", i, original[i], reshaped[i])
		}
	}
	if math.Abs(deltaC) > 1e-12 {
		t.Fatalf("Long2Short Δ over C = %v, want 0", deltaC)
	}

	// The zero-sum delta is a property of the reshaped rewards themselves and is
	// therefore preserved no matter which advantage normalization mgpo applies
	// next. Confirm both paths accept the reshaped group without error and that
	// the no-std path does not reintroduce a nonzero net shift over C relative to
	// the std path's own group-mean-zero advantage. (Both advantage forms are
	// mean-zero within a group by construction.)
	group := [][]float64{reshaped}
	for _, opt := range []Options{{}, {DrGRPOAdvantage: true}} {
		adv, err := ScaledAdvantagesOpt(group, 0, opt)
		if err != nil {
			t.Fatalf("ScaledAdvantagesOpt opt=%+v: %v", opt, err)
		}
		var sum float64
		for _, a := range adv[0] {
			sum += a
		}
		if math.Abs(sum) > 1e-9 {
			t.Fatalf("opt=%+v: advantage sum over group = %v, want ~0 (mean-zero)", opt, sum)
		}
	}
}

// toyRollouts builds a small group of per-token log-prob tensors whose
// importance ratio exp(current-old) clears 1+ε on the high side, so the upper
// clip genuinely binds and the ε_low/ε_high distinction is observable. The drift
// is ~+0.35 per token (ratio ≈ 1.42 > 1.28 > 1.25), straddling both clip
// ceilings.
func toyRollouts(t *testing.T) (current, old, ref, mask *mlx.Array) {
	t.Helper()
	const seqs, toks = 4, 3
	mk := func(seed float32) *mlx.Array {
		vals := make([]float32, seqs*toks)
		for i := range vals {
			vals[i] = seed + float32(i)*0.02
		}
		return mlx.NewArray(vals, seqs, toks)
	}
	current = mk(-0.2)
	old = mlx.StopGradient(mk(-0.55)) // current - old ≈ +0.35 ⇒ ratio ≈ 1.42
	ref = mlx.StopGradient(mk(-0.5))
	maskVals := make([]float32, seqs*toks)
	for i := range maskVals {
		maskVals[i] = 1
	}
	mask = mlx.NewArray(maskVals, seqs, toks)
	return current, old, ref, mask
}
