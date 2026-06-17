package clr

import (
	"context"
	"math"
	"strings"
	"testing"
)

// Invariant (DESIGN §4.7, §5.6): r_k = 1 iff all M claims valid.
func TestReliabilityOneIffAllValid(t *testing.T) {
	allValid := Trajectory{Answer: "x", Claims: []int{1, 1, 1, 1, 1}}
	r, err := Reliability(allValid)
	if err != nil {
		t.Fatalf("Reliability: %v", err)
	}
	if r != 1 {
		t.Fatalf("all-valid r_k = %v, want 1", r)
	}
	// Any single invalid claim makes r_k < 1.
	for i := 0; i < 5; i++ {
		claims := []int{1, 1, 1, 1, 1}
		claims[i] = 0
		r, err := Reliability(Trajectory{Answer: "x", Claims: claims})
		if err != nil {
			t.Fatalf("Reliability: %v", err)
		}
		if r >= 1 {
			t.Fatalf("one invalid claim at %d still gave r_k = %v, want < 1", i, r)
		}
	}
}

// Invariant (DESIGN §4.7): M=5, 4/5 valid ⇒ 0.8^5 ≈ 0.328, within 1e-3.
func TestReliabilityFourOfFive(t *testing.T) {
	r, err := Reliability(Trajectory{Answer: "x", Claims: []int{1, 1, 1, 1, 0}})
	if err != nil {
		t.Fatalf("Reliability: %v", err)
	}
	want := math.Pow(0.8, 5) // 0.32768
	if math.Abs(r-want) > 1e-3 {
		t.Fatalf("4/5-valid r_k = %v, want %v (±1e-3)", r, want)
	}
	if math.Abs(r-0.32768) > 1e-5 {
		t.Fatalf("4/5-valid r_k = %v, want 0.32768", r)
	}
}

// Invariant (DESIGN §4.7): one invalid claim drops r_k sharply — far more than
// a linear 1/M penalty would.
func TestReliabilityDropsSharply(t *testing.T) {
	full, _ := Reliability(Trajectory{Answer: "x", Claims: []int{1, 1, 1, 1, 1}})
	one, _ := Reliability(Trajectory{Answer: "x", Claims: []int{1, 1, 1, 1, 0}})
	drop := full - one // 1 - 0.32768 = 0.67232
	linear := 1.0 / 5  // a single-claim linear penalty would be 0.2
	if drop <= linear {
		t.Fatalf("drop %v is not sharper than linear penalty %v", drop, linear)
	}
	// And monotone: more invalid claims => strictly lower reliability.
	prev := full
	for k := 1; k <= 5; k++ {
		claims := make([]int, 5)
		for i := 0; i < 5; i++ {
			if i >= k {
				claims[i] = 1
			}
		}
		r, _ := Reliability(Trajectory{Answer: "x", Claims: claims})
		if r >= prev {
			t.Fatalf("reliability not monotone decreasing: k=%d r=%v prev=%v", k, r, prev)
		}
		prev = r
	}
}

// reliability core table: closed-form ((valid/M)^M).
func TestReliabilityCore(t *testing.T) {
	tests := []struct {
		claims []int
		want   float64
	}{
		{[]int{1, 1, 1, 1, 1}, 1},
		{[]int{1, 1, 1, 1, 0}, math.Pow(0.8, 5)},
		{[]int{1, 1, 1, 0, 0}, math.Pow(0.6, 5)},
		{[]int{0, 0, 0, 0, 0}, 0},
		{[]int{1}, 1},
		{[]int{1, 0}, 0.25}, // (1/2)^2
	}
	for _, tt := range tests {
		got, err := reliability(tt.claims)
		if err != nil {
			t.Fatalf("reliability(%v): %v", tt.claims, err)
		}
		if math.Abs(got-tt.want) > 1e-12 {
			t.Fatalf("reliability(%v) = %v, want %v", tt.claims, got, tt.want)
		}
	}
}

func TestReliabilityValidation(t *testing.T) {
	if _, err := reliability(nil); err == nil {
		t.Fatal("want error for empty claims")
	}
	if _, err := reliability([]int{1, 2, 0}); err == nil {
		t.Fatal("want error for verdict not in {0,1}")
	}
}

// selectAnswer picks the cluster with maximal summed reliability, not the
// largest cluster: a single fully-reliable trajectory beats a majority of
// flawed ones.
func TestSelectByReliabilityNotMajority(t *testing.T) {
	trajs := []Trajectory{
		// Three "wrong" trajectories, each only 3/5 valid -> r ≈ 0.0778 each.
		{Answer: "wrong", Claims: []int{1, 1, 1, 0, 0}},
		{Answer: "wrong", Claims: []int{1, 1, 1, 0, 0}},
		{Answer: "wrong", Claims: []int{1, 1, 1, 0, 0}},
		// One "right" trajectory, fully valid -> r = 1.
		{Answer: "right", Claims: []int{1, 1, 1, 1, 1}},
	}
	ans, sum, err := selectAnswer(trajs, nil)
	if err != nil {
		t.Fatalf("selectAnswer: %v", err)
	}
	if ans != "right" {
		t.Fatalf("selected %q (sum %v), want \"right\" (reliability beats majority)", ans, sum)
	}
	// Sanity: the wrong cluster's summed reliability is below the right one.
	wrongR := 3 * math.Pow(0.6, 5)
	if sum <= wrongR {
		t.Fatalf("winning sum %v not above wrong cluster sum %v", sum, wrongR)
	}
}

// Clustering uses the supplied Equivalence; answers that normalize equal merge.
func TestSelectUsesEquivalence(t *testing.T) {
	eq := func(a, b string) bool {
		return strings.TrimSpace(strings.ToLower(a)) == strings.TrimSpace(strings.ToLower(b))
	}
	trajs := []Trajectory{
		{Answer: "42", Claims: []int{1, 1, 0, 0, 0}},   // r ≈ 0.0102
		{Answer: " 42 ", Claims: []int{1, 1, 1, 1, 1}}, // r = 1, merges with "42"
		{Answer: "7", Claims: []int{1, 1, 1, 1, 1}},    // r = 1, separate
	}
	ans, sum, err := selectAnswer(trajs, eq)
	if err != nil {
		t.Fatalf("selectAnswer: %v", err)
	}
	if strings.TrimSpace(ans) != "42" {
		t.Fatalf("selected %q, want 42 (merged cluster wins)", ans)
	}
	if sum <= 1 {
		t.Fatalf("merged cluster sum %v should exceed 1 (1 + 0.0102...)", sum)
	}
}

// Ties are broken by first appearance for determinism.
func TestSelectTieDeterministic(t *testing.T) {
	trajs := []Trajectory{
		{Answer: "a", Claims: []int{1, 1, 1, 1, 1}},
		{Answer: "b", Claims: []int{1, 1, 1, 1, 1}},
	}
	for i := 0; i < 10; i++ {
		ans, _, err := selectAnswer(trajs, nil)
		if err != nil {
			t.Fatalf("selectAnswer: %v", err)
		}
		if ans != "a" {
			t.Fatalf("tie broke to %q, want first-appearance \"a\"", ans)
		}
	}
}

// Score over R flows with a deterministic fake: a fully-reliable correct
// trajectory present in every flow yields mean Pass@1 = 1.
func TestScoreAllCorrect(t *testing.T) {
	v := FakeVerifier{Trajs: []Trajectory{
		{Answer: "gold", Claims: []int{1, 1, 1, 1, 1}},
		{Answer: "other", Claims: []int{1, 1, 0, 0, 0}},
	}}
	res, err := Score(context.Background(), v, "q", "gold", DefaultK, DefaultM, DefaultR, nil, nil)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if res.MeanPass != 1 {
		t.Fatalf("MeanPass = %v, want 1", res.MeanPass)
	}
	if len(res.Answers) != DefaultR {
		t.Fatalf("got %d flow answers, want %d", len(res.Answers), DefaultR)
	}
}

// When the reliable cluster is the wrong answer, mean Pass@1 = 0.
func TestScoreAllWrong(t *testing.T) {
	v := FakeVerifier{Trajs: []Trajectory{
		{Answer: "bad", Claims: []int{1, 1, 1, 1, 1}}, // reliable but wrong
		{Answer: "gold", Claims: []int{1, 0, 0, 0, 0}},
	}}
	res, err := Score(context.Background(), v, "q", "gold", 8, 5, 4, nil, nil)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if res.MeanPass != 0 {
		t.Fatalf("MeanPass = %v, want 0", res.MeanPass)
	}
}

func TestScoreValidation(t *testing.T) {
	ctx := context.Background()
	v := FakeVerifier{Trajs: []Trajectory{{Answer: "x", Claims: []int{1}}}}
	if _, err := Score(ctx, nil, "q", "g", 1, 1, 1, nil, nil); err == nil {
		t.Fatal("want error for nil verifier")
	}
	for _, bad := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}} {
		if _, err := Score(ctx, v, "q", "g", bad[0], bad[1], bad[2], nil, nil); err == nil {
			t.Fatalf("want error for k,m,r = %v", bad)
		}
	}
}

// A verifier that returns a malformed shape is a hard error.
func TestScoreShapeMismatch(t *testing.T) {
	bad := badVerifier{}
	_, err := Score(context.Background(), bad, "q", "g", 4, 5, 2, nil, nil)
	if err == nil {
		t.Fatal("want error for trajectory count mismatch")
	}
}

type badVerifier struct{}

func (badVerifier) Trajectories(_ context.Context, _ string, k, m int) ([]Trajectory, error) {
	// Returns one fewer trajectory than requested.
	out := make([]Trajectory, k-1)
	for i := range out {
		out[i] = Trajectory{Answer: "x", Claims: make([]int, m)}
	}
	return out, nil
}
