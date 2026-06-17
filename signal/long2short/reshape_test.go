package long2short

import (
	"math"
	"testing"
)

func sum(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s
}

// Invariant (DESIGN §5.3): the reshape is zero-sum over the correct set C —
// Σ_C (r′ᵢ − rᵢ) = 0 — so the group mean reward is unchanged.
func TestZeroSumOverCorrectSet(t *testing.T) {
	traces := []Trace{
		{Reward: 1, Length: 10, Correct: true},
		{Reward: 1, Length: 20, Correct: true},
		{Reward: 1, Length: 40, Correct: true},
		{Reward: 0, Length: 5, Correct: false}, // incorrect: untouched
	}
	out, err := Reshape(traces, DefaultLambda)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	// Delta only over the correct set; incorrect reward unchanged.
	if out[3] != 0 {
		t.Fatalf("incorrect trace reward changed: %v", out[3])
	}
	var delta float64
	for i := range traces {
		if traces[i].Correct {
			delta += out[i] - traces[i].Reward
		}
	}
	if math.Abs(delta) > 1e-12 {
		t.Fatalf("Σ_C (r'-r) = %v, want 0", delta)
	}
	// Group mean unchanged.
	before := sum([]float64{1, 1, 1, 0}) / 4
	after := sum(out) / 4
	if math.Abs(before-after) > 1e-12 {
		t.Fatalf("group mean changed: %v -> %v", before, after)
	}
}

// Invariant (DESIGN §5.3): equal-length correct set ⇒ no-op.
func TestEqualLengthNoOp(t *testing.T) {
	traces := []Trace{
		{Reward: 1, Length: 30, Correct: true},
		{Reward: 1, Length: 30, Correct: true},
		{Reward: 1, Length: 30, Correct: true},
	}
	out, err := Reshape(traces, DefaultLambda)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	for i := range out {
		if out[i] != traces[i].Reward {
			t.Fatalf("equal-length should be no-op: out[%d]=%v", i, out[i])
		}
	}
}

// Shorter correct traces are nudged above the mean, longer below; the per-trace
// shift matches the closed form λ·(s−s̄)/max|s−s̄|.
func TestBrevityDirectionAndMagnitude(t *testing.T) {
	traces := []Trace{
		{Reward: 1, Length: 10, Correct: true}, // s = 0.1 (shortest -> highest brevity)
		{Reward: 1, Length: 100, Correct: true},
	}
	const lambda = 0.2
	out, err := Reshape(traces, lambda)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	// Shorter trace should be rewarded more than longer.
	if !(out[0] > traces[0].Reward && out[1] < traces[1].Reward) {
		t.Fatalf("brevity direction wrong: out=%v", out)
	}
	// Closed form: s = [0.1, 0.01], mean = 0.055, dev = [0.045, -0.045],
	// maxDev = 0.045, shift = ±λ.
	if math.Abs((out[0]-traces[0].Reward)-lambda) > 1e-9 {
		t.Fatalf("short-trace shift = %v, want %v", out[0]-traces[0].Reward, lambda)
	}
	if math.Abs((out[1]-traces[1].Reward)+lambda) > 1e-9 {
		t.Fatalf("long-trace shift = %v, want %v", out[1]-traces[1].Reward, -lambda)
	}
}

// The max in the denominator is over the correct set C only — an incorrect
// trace, however short, must not influence the normalization.
func TestDenominatorOverCorrectSetOnly(t *testing.T) {
	// Two correct traces; an incorrect trace far shorter than either.
	withIncorrect := []Trace{
		{Reward: 1, Length: 10, Correct: true},
		{Reward: 1, Length: 20, Correct: true},
		{Reward: 0, Length: 1, Correct: false},
	}
	withoutIncorrect := []Trace{
		{Reward: 1, Length: 10, Correct: true},
		{Reward: 1, Length: 20, Correct: true},
	}
	a, _ := Reshape(withIncorrect, DefaultLambda)
	b, _ := Reshape(withoutIncorrect, DefaultLambda)
	// The correct-set shifts must be identical regardless of the incorrect
	// trace's presence.
	if math.Abs(a[0]-b[0]) > 1e-12 || math.Abs(a[1]-b[1]) > 1e-12 {
		t.Fatalf("incorrect trace influenced normalization: with=%v without=%v", a[:2], b)
	}
}

func TestEmptyAndSingletonCorrectSetNoOp(t *testing.T) {
	// No correct traces.
	none := []Trace{{Reward: 0, Length: 10, Correct: false}}
	out, _ := Reshape(none, DefaultLambda)
	if out[0] != 0 {
		t.Fatalf("no-correct should be no-op, got %v", out)
	}
	// Single correct trace.
	one := []Trace{{Reward: 1, Length: 10, Correct: true}, {Reward: 0, Length: 5, Correct: false}}
	out2, _ := Reshape(one, DefaultLambda)
	if out2[0] != 1 || out2[1] != 0 {
		t.Fatalf("singleton-correct should be no-op, got %v", out2)
	}
}

func TestReshapeValidation(t *testing.T) {
	// A non-positive length on any correct trace is an error (two correct
	// traces so the length check is reached, not the singleton short-circuit).
	bad := []Trace{{Reward: 1, Length: 10, Correct: true}, {Reward: 1, Length: 0, Correct: true}}
	if _, err := Reshape(bad, DefaultLambda); err == nil {
		t.Fatal("want error for non-positive correct-trace length")
	}
	if _, err := Reshape(nil, -1); err == nil {
		t.Fatal("want error for negative lambda")
	}
}

// Lambda = 0 is a no-op even with varied lengths.
func TestZeroLambdaNoOp(t *testing.T) {
	traces := []Trace{
		{Reward: 1, Length: 10, Correct: true},
		{Reward: 1, Length: 50, Correct: true},
	}
	out, _ := Reshape(traces, 0)
	for i := range out {
		if out[i] != traces[i].Reward {
			t.Fatalf("λ=0 should be no-op: out=%v", out)
		}
	}
}
