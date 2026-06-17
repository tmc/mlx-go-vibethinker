package mgpo

import (
	"math"
	"testing"
)

// Phase-B property (DESIGN_RL_UPGRADE.md §2 Tier 2, DCPO-SAS): with no history
// store (nil stats) ScaledAdvantagesStep is bit-identical to today's
// ScaledAdvantagesOpt for every λ — SAS is fully off-path by default.
// Additionally the first visit over a fresh store is bit-identical, because SAS
// initializes A_total to A_new and both smoothed estimates collapse to A_new.
// The control half (multiple accumulated visits) must genuinely differ, proving
// the knob is live.
func TestDCPOSASFirstVisitBitIdenticalToBaseline(t *testing.T) {
	rewards := [][]float64{
		{1, 0, 1, 1},
		{0, 0, 1, 0},
		{1, 1, 1, 1}, // degenerate group (std=0)
	}
	ids := []string{"q0", "q1", "q2"}
	for _, lambda := range []float64{0, 0.5, 1.0, 4.0} {
		for _, opts := range []Options{{}, {DrGRPOAdvantage: true}} {
			want, err := ScaledAdvantagesOpt(rewards, lambda, opts)
			if err != nil {
				t.Fatalf("ScaledAdvantagesOpt: %v", err)
			}

			// nil stats: SAS off entirely.
			gotNil, err := ScaledAdvantagesStep(rewards, lambda, opts, nil, nil)
			if err != nil {
				t.Fatalf("ScaledAdvantagesStep nil stats: %v", err)
			}
			assertEqualGroups(t, gotNil, want, "nil-stats SAS off")

			// First visit over a fresh store: also bit-identical.
			fresh := NewPromptStats()
			gotFirst, err := ScaledAdvantagesStep(rewards, lambda, opts, fresh, ids)
			if err != nil {
				t.Fatalf("ScaledAdvantagesStep first visit: %v", err)
			}
			assertEqualGroups(t, gotFirst, want, "first-visit SAS")
		}
	}

	// Control: an accumulated store over several steps with varying rewards must
	// differ from the per-step baseline once i ≥ 3 (the (i−1)/i vs 1/i weights
	// only diverge there). Use the std-normalized path on a single prompt.
	stats := NewPromptStats()
	steps := [][][]float64{
		{{1, 0, 1, 1}}, // visit 1
		{{0, 1, 0, 0}}, // visit 2
		{{1, 1, 0, 1}}, // visit 3 — asymmetric weights now bite
	}
	var smoothed, plain [][]float64
	for _, step := range steps {
		var err error
		smoothed, err = ScaledAdvantagesStep(step, 0, Options{}, stats, []string{"q"})
		if err != nil {
			t.Fatalf("ScaledAdvantagesStep step: %v", err)
		}
		plain, err = ScaledAdvantagesOpt(step, 0, Options{})
		if err != nil {
			t.Fatalf("ScaledAdvantagesOpt step: %v", err)
		}
	}
	differs := false
	for j := range smoothed[0] {
		if math.Abs(smoothed[0][j]-plain[0][j]) > 1e-12 {
			differs = true
		}
	}
	if !differs {
		t.Fatalf("DCPO-SAS on visit 3 produced %v identical to plain advantage %v; knob is inert", smoothed[0], plain[0])
	}
}

// Phase-B property: SAS picks the minimum-magnitude of the two smoothed
// estimates per rollout slot, exactly as the paper specifies. We drive a known
// history and check the closed form by hand.
func TestDCPOSASMinMagnitudeSelection(t *testing.T) {
	// Drive raw advantages directly through the unexported smoother so the
	// arithmetic is exact and independent of GroupAdvantage.
	s := NewPromptStats()

	// Visit 1: A_total := A_new = [2, -2].
	v1, err := s.smooth("q", []float64{2, -2})
	if err != nil {
		t.Fatalf("smooth v1: %v", err)
	}
	if v1[0] != 2 || v1[1] != -2 {
		t.Fatalf("visit 1 = %v, want [2 -2] (first visit == plain)", v1)
	}

	// Visit 2: i=2, weights 1/2 and 1/2 ⇒ SA_new == SA_total == 0.5*A_new+0.5*A_total.
	// A_new = [0, 0]; A_total = [2, -2] ⇒ both estimates = [1, -1]; A = [1, -1].
	v2, err := s.smooth("q", []float64{0, 0})
	if err != nil {
		t.Fatalf("smooth v2: %v", err)
	}
	if math.Abs(v2[0]-1) > 1e-12 || math.Abs(v2[1]+1) > 1e-12 {
		t.Fatalf("visit 2 = %v, want [1 -1]", v2)
	}

	// Visit 3: i=3, A_new = [3, 3], A_total = [1, -1].
	//   slot0: SA_new = (2/3)*3 + (1/3)*1 = 2.3333; SA_total = (1/3)*3 + (2/3)*1 = 1.6667
	//          |1.6667| < |2.3333| ⇒ A = 1.6667
	//   slot1: SA_new = (2/3)*3 + (1/3)*(-1) = 1.6667; SA_total = (1/3)*3 + (2/3)*(-1) = 0.3333
	//          |0.3333| < |1.6667| ⇒ A = 0.3333
	v3, err := s.smooth("q", []float64{3, 3})
	if err != nil {
		t.Fatalf("smooth v3: %v", err)
	}
	want0 := 1.0/3.0*3 + 2.0/3.0*1
	want1 := 1.0/3.0*3 + 2.0/3.0*(-1)
	if math.Abs(v3[0]-want0) > 1e-12 {
		t.Fatalf("visit 3 slot 0 = %v, want min-magnitude %v", v3[0], want0)
	}
	if math.Abs(v3[1]-want1) > 1e-12 {
		t.Fatalf("visit 3 slot 1 = %v, want min-magnitude %v", v3[1], want1)
	}
}

// Phase-B cross-cutting invariant: the MGPO no-op rule survives SAS — w_ME
// multiplies the smoothed advantage, never the raw reward. At λ=0 the result is
// exactly the smoothed advantage; at λ>0 it is exactly w_ME(p_c)·(smoothed A)
// per group.
func TestDCPOSASNoOpRulePreserved(t *testing.T) {
	const lambda = 3.0
	id := []string{"q"}

	// Build the smoothed advantage independently via the unexported smoother on
	// a parallel store driven with the same advantage sequence.
	statsA := NewPromptStats()
	statsB := NewPromptStats()
	// Two visits so SAS is non-trivial.
	steps := [][][]float64{{{1, 0, 1, 1}}, {{0, 1, 1, 1}}}
	var lastScaled, lastSmoothedAdv [][]float64
	for _, step := range steps {
		scaled, err := ScaledAdvantagesStep(step, lambda, Options{}, statsA, id)
		if err != nil {
			t.Fatalf("ScaledAdvantagesStep: %v", err)
		}
		lastScaled = scaled
		// Reconstruct: raw group advantage → smooth → expect w_ME * smoothed.
		rawAdv := Options{}.groupAdvantage(step)
		sm, err := statsB.smooth("q", rawAdv[0])
		if err != nil {
			t.Fatalf("smooth: %v", err)
		}
		lastSmoothedAdv = [][]float64{sm}
	}
	w, err := Weight(lambda, Accuracy(steps[len(steps)-1][0]))
	if err != nil {
		t.Fatalf("Weight: %v", err)
	}
	for j := range lastScaled[0] {
		want := w * lastSmoothedAdv[0][j]
		if math.Abs(lastScaled[0][j]-want) > 1e-12 {
			t.Fatalf("slot %d: %v != w_ME·(smoothed A) %v — no-op rule broken", j, lastScaled[0][j], want)
		}
	}
}

func assertEqualGroups(t *testing.T, got, want [][]float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d groups != %d", label, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s: group %d size %d != %d", label, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("%s: group %d slot %d: %v != %v (must be bit-identical)", label, i, j, got[i][j], want[i][j])
			}
		}
	}
}
