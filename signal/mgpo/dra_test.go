package mgpo

import (
	"math"
	"testing"
)

// Phase-C property (DESIGN_RL_UPGRADE.md §2 Tier 3, DRA-GRPO): off-path
// bit-identical. DRA is opt-in — when the reweight is simply not applied, the
// advantage is the unmodified baseline. The control half (a diverse embedder)
// must change the reweighted rewards, and hence the advantage, proving the knob
// is live.
func TestDRAOffPathAndLiveKnob(t *testing.T) {
	rewards := []float64{1, 1, 0, 1}

	// Off-path: not invoking DRA leaves the advantage at baseline. (Trivially
	// true structurally, but we pin it: the reward vector is untouched.)
	baseAdv, err := ScaledAdvantagesOpt([][]float64{rewards}, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt baseline: %v", err)
	}

	// Live knob: diverse rollout texts produce a non-uniform divisor, so the
	// reweighted rewards differ from the raw rewards and the advantage moves.
	diverse := []string{
		"the derivative is two x",
		"by induction the sum telescopes",
		"factor the quadratic and solve",
		"apply the chain rule carefully here",
	}
	rw, err := DiversityReweight(rewards, diverse, FakeEmbedder{})
	if err != nil {
		t.Fatalf("DiversityReweight diverse: %v", err)
	}
	changed := false
	for i := range rewards {
		if math.Abs(rw[i]-rewards[i]) > 1e-9 {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("DRA left rewards unchanged on diverse rollouts %v; diversity knob is inert", rw)
	}
	draAdv, err := ScaledAdvantagesOpt([][]float64{rw}, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt DRA: %v", err)
	}
	advMoved := false
	for j := range baseAdv[0] {
		if math.Abs(draAdv[0][j]-baseAdv[0][j]) > 1e-9 {
			advMoved = true
		}
	}
	if !advMoved {
		t.Fatalf("DRA reweight did not move the advantage; knob is inert under std normalization")
	}
}

// Phase-C property: an identity embedder is the DRA no-op. When every rollout is
// identical, all pairwise similarities are 1 and the divisor is the constant G
// for every rollout, so the reweight is a uniform rescale R_i → R_i/G.
//
// A uniform rescale of the rewards cancels under group-relative advantage,
// exactly in the Dr.GRPO no-std path (advantage is r−mean, so r/G−mean/G =
// (r−mean)/G — the advantage is uniformly scaled by 1/G and, after std-style
// normalization it would cancel entirely). The std-normalized path is the same
// up to the substrate's fixed std+eps regularizer (eps=1e-4): dividing the
// rewards by G shrinks std by G but not eps, a deliberate substrate guard, so
// the cancellation is exact only in the eps→0 limit. We therefore assert the
// clean Dr.GRPO identity exactly and the std-path identity within the eps-scale
// tolerance — documenting that DRA's "identity == baseline" is a property of the
// uniform rescale, not of the eps regularizer.
func TestDRAIdentityEmbedderEqualsBaseline(t *testing.T) {
	rewards := []float64{1, 0, 1, 0}
	identical := []string{"same trace", "same trace", "same trace", "same trace"}

	rw, err := DiversityReweight(rewards, identical, FakeEmbedder{})
	if err != nil {
		t.Fatalf("DiversityReweight identical: %v", err)
	}
	// Every divisor is G = 4, so each reward is exactly r/4.
	for i := range rewards {
		if math.Abs(rw[i]-rewards[i]/4) > 1e-12 {
			t.Fatalf("identical rollout %d divisor != G: rw=%v want %v", i, rw[i], rewards[i]/4)
		}
	}

	// Dr.GRPO no-std path: the reweighted advantage is exactly 1/G times the
	// baseline advantage (uniform rescale cancels up to the constant factor).
	baseDr := Options{DrGRPOAdvantage: true}.groupAdvantage([][]float64{rewards})[0]
	drDr := Options{DrGRPOAdvantage: true}.groupAdvantage([][]float64{rw})[0]
	for j := range baseDr {
		if math.Abs(drDr[j]-baseDr[j]/4) > 1e-12 {
			t.Fatalf("Dr.GRPO identity slot %d = %v, want baseline/G = %v", j, drDr[j], baseDr[j]/4)
		}
	}

	// Std-normalized path: equals the baseline up to the eps regularizer. The
	// substrate's std+eps with eps=1e-4 means a ~1e-4-scale deviation, no more.
	baseAdv, err := ScaledAdvantagesOpt([][]float64{rewards}, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt baseline: %v", err)
	}
	draAdv, err := ScaledAdvantagesOpt([][]float64{rw}, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt DRA: %v", err)
	}
	for j := range baseAdv[0] {
		if math.Abs(draAdv[0][j]-baseAdv[0][j]) > 1e-3 {
			t.Fatalf("std-path identity slot %d = %v deviates from baseline %v beyond the eps scale", j, draAdv[0][j], baseAdv[0][j])
		}
	}
}

// Phase-C property: a distinctive rollout keeps more reward than a crowded one.
// With three near-identical rollouts and one distinctive rollout of equal raw
// reward, the distinctive rollout's reweighted reward must exceed the crowded
// ones' — DRA down-weights redundant modes.
func TestDRADownWeightsCrowdedModes(t *testing.T) {
	rewards := []float64{1, 1, 1, 1}
	texts := []string{
		"aaaa aaaa aaaa", // three near-identical crowded rollouts
		"aaaa aaaa aaaa",
		"aaaa aaaa aaaa",
		"zzzz qqqq wwww", // one distinctive rollout
	}
	rw, err := DiversityReweight(rewards, texts, FakeEmbedder{})
	if err != nil {
		t.Fatalf("DiversityReweight: %v", err)
	}
	// The distinctive rollout (index 3) should retain more reward than a crowded
	// one (index 0), since its similarity sum is smaller.
	if !(rw[3] > rw[0]) {
		t.Fatalf("DRA did not favor the distinctive rollout: crowded=%v distinctive=%v", rw[0], rw[3])
	}
}

// Phase-C cross-cutting invariant: the DRA reweight happens before advantage
// normalization, so w_ME still multiplies the resulting advantage — the no-op
// rule is preserved. At λ=0 the result is exactly the group advantage of the
// reweighted rewards; at λ>0 it is exactly w_ME(p_c)·that advantage, with p_c
// computed from the ORIGINAL rewards (reweighting changes magnitudes, not the
// pass/fail accuracy).
func TestDRANoOpRulePreserved(t *testing.T) {
	const lambda = 2.0
	rewards := []float64{1, 0, 1, 1} // p_c = 0.75
	texts := []string{"alpha path", "beta path", "gamma path", "delta path"}

	rw, err := DiversityReweight(rewards, texts, FakeEmbedder{})
	if err != nil {
		t.Fatalf("DiversityReweight: %v", err)
	}
	// Feed reweighted rewards through the standard pipeline.
	got, err := ScaledAdvantagesOpt([][]float64{rw}, lambda, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt: %v", err)
	}
	// Reconstruct: group advantage of reweighted rewards, scaled by w_ME(p_c)
	// where p_c is from the original rewards (DRA preserves pass/fail labels —
	// dividing a positive reward by a positive divisor keeps it positive).
	rawAdv := Options{}.groupAdvantage([][]float64{rw})[0]
	w, err := Weight(lambda, Accuracy(rewards))
	if err != nil {
		t.Fatalf("Weight: %v", err)
	}
	for j := range rw {
		want := w * rawAdv[j]
		if math.Abs(got[0][j]-want) > 1e-12 {
			t.Fatalf("slot %d: %v != w_ME·adv(reweighted) %v — no-op rule broken", j, got[0][j], want)
		}
	}
	// And DRA must not have flipped any pass/fail label: accuracy is unchanged.
	if Accuracy(rw) != Accuracy(rewards) {
		t.Fatalf("DRA changed group accuracy: %v != %v", Accuracy(rw), Accuracy(rewards))
	}
}
