package distill

import (
	"context"
	"math"
	"testing"
)

// Invariant (DESIGN §5.8): S_LP ranks a deliberately-mispredicted trace above a
// well-modeled one. The well-modeled trace has log-probs near 0 (high prob);
// the mispredicted trace has very negative log-probs (low prob).
func TestScoreRanksMispredictedAbove(t *testing.T) {
	wellModeled := []float64{-0.05, -0.02, -0.1, -0.03} // student is confident & right
	mispredicted := []float64{-5, -6, -4.5, -7}         // student assigns tiny prob
	sw, err := Score(wellModeled)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	sm, err := Score(mispredicted)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !(sm > sw) {
		t.Fatalf("S_LP should rank mispredicted (%.3f) above well-modeled (%.3f)", sm, sw)
	}
	// S_LP is non-negative for log-probs ≤ 0 and equals the mean NLL.
	if sw < 0 || sm < 0 {
		t.Fatalf("S_LP should be non-negative: sw=%v sm=%v", sw, sm)
	}
}

// S_LP is length-normalized: duplicating a trace's tokens does not change it.
func TestScoreLengthNormalized(t *testing.T) {
	base := []float64{-1, -2, -3}
	doubled := append(append([]float64{}, base...), base...)
	s1, _ := Score(base)
	s2, _ := Score(doubled)
	if math.Abs(s1-s2) > 1e-12 {
		t.Fatalf("length normalization broken: %v vs %v", s1, s2)
	}
	// Closed form: mean NLL = -(−1−2−3)/3 = 2.
	if math.Abs(s1-2) > 1e-12 {
		t.Fatalf("S_LP = %v, want 2", s1)
	}
}

func TestScoreEmpty(t *testing.T) {
	if _, err := Score(nil); err == nil {
		t.Fatal("want error for empty log-probs")
	}
}

// fakeScorer returns canned per-token log-probs keyed by trace length.
type fakeScorer struct{ lp []float64 }

func (f fakeScorer) LogProbs(ctx context.Context, tokens []int) ([]float64, error) {
	return f.lp, nil
}

func TestScoreTraceViaScorer(t *testing.T) {
	s := fakeScorer{lp: []float64{-1, -1, -1, -1}}
	got, err := ScoreTrace(context.Background(), s, []int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatalf("ScoreTrace: %v", err)
	}
	if math.Abs(got-1) > 1e-12 {
		t.Fatalf("S_LP = %v, want 1", got)
	}
}

func TestSelectDropsShortTraces(t *testing.T) {
	traces := []Trace{
		{ID: "a", Domain: "math", Length: 10, Score: 5}, // dropped (< MinLength 64)
		{ID: "b", Domain: "math", Length: 100, Score: 2},
		{ID: "c", Domain: "math", Length: 200, Score: 3},
	}
	out := Select(traces, SelectParams{MinLength: 64, Buckets: 1, OutlierQuantile: 0})
	for _, tr := range out {
		if tr.ID == "a" {
			t.Fatal("short trace not dropped")
		}
	}
	if len(out) != 2 {
		t.Fatalf("kept %d, want 2", len(out))
	}
}

func TestSelectTrimsHighScoreOutliers(t *testing.T) {
	// One bucket, 10 traces; trim top 20% (2 highest scores).
	var traces []Trace
	for i := range 10 {
		traces = append(traces, Trace{
			ID:     string(rune('a' + i)),
			Domain: "math",
			Length: 100,
			Score:  float64(i), // scores 0..9
		})
	}
	out := Select(traces, SelectParams{MinLength: 1, Buckets: 1, OutlierQuantile: 0.2})
	if len(out) != 8 {
		t.Fatalf("kept %d, want 8 (trimmed 2 outliers)", len(out))
	}
	// The two highest scores (8, 9) must be gone.
	for _, tr := range out {
		if tr.Score >= 8 {
			t.Fatalf("high-score outlier %v not trimmed", tr.Score)
		}
	}
	// Output sorted by descending score within the bucket.
	for i := 1; i < len(out); i++ {
		if out[i].Score > out[i-1].Score {
			t.Fatalf("output not score-descending at %d", i)
		}
	}
}

// Selection is per domain: bucketing uses each domain's own length scale, and
// the result mixes across domains.
func TestSelectPerDomainBuckets(t *testing.T) {
	traces := []Trace{
		// math: short scale (100..160)
		{ID: "m1", Domain: "math", Length: 100, Score: 1},
		{ID: "m2", Domain: "math", Length: 160, Score: 2},
		// code: long scale (1000..4000) — must not share math's buckets
		{ID: "c1", Domain: "code", Length: 1000, Score: 1},
		{ID: "c2", Domain: "code", Length: 4000, Score: 2},
	}
	out := Select(traces, SelectParams{MinLength: 1, Buckets: 2, OutlierQuantile: 0})
	if len(out) != 4 {
		t.Fatalf("kept %d, want 4 (all, mixed across domains)", len(out))
	}
	// Both domains represented.
	seen := map[string]bool{}
	for _, tr := range out {
		seen[tr.Domain] = true
	}
	if !seen["math"] || !seen["code"] {
		t.Fatalf("domains not mixed: %v", seen)
	}
}

func TestSelectNeverEmptiesNonEmptyBucket(t *testing.T) {
	// A single trace with an aggressive quantile must still keep one.
	out := Select([]Trace{{ID: "x", Domain: "math", Length: 100, Score: 5}},
		SelectParams{MinLength: 1, Buckets: 1, OutlierQuantile: 0.99})
	if len(out) != 1 {
		t.Fatalf("kept %d, want 1 (never trim a bucket to empty)", len(out))
	}
}
