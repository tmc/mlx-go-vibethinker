package data

import (
	"math"
	"testing"
)

// Invariant (DESIGN §4.6): rep_n = 0 for text with all-distinct n-grams, and
// rep_n → 1 for a single token repeated; short text (< n tokens) is 0.
func TestRepetitionRateBounds(t *testing.T) {
	tests := []struct {
		name string
		text string
		n    int
		want float64
	}{
		{"all distinct unigrams", "a b c d e", 1, 0},
		{"all distinct bigrams", "a b c d e", 2, 0},
		// "a a a a a": one distinct unigram out of 5 total ⇒ 1 - 1/5 = 0.8.
		{"repeated unigram", "a a a a a", 1, 0.8},
		// bigrams of "a a a a a": 4 total, all "a a" ⇒ 1 - 1/4 = 0.75.
		{"repeated bigram", "a a a a a", 2, 0.75},
		{"empty text", "", 3, 0},
		{"fewer tokens than n", "a b", 3, 0},
		{"exactly n tokens", "a b c", 3, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RepetitionRate(tt.text, tt.n)
			if err != nil {
				t.Fatalf("RepetitionRate: %v", err)
			}
			if math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("rep = %v, want %v", got, tt.want)
			}
		})
	}
}

// As a degenerate text grows, the single-token repetition rate approaches 1.
func TestRepetitionRateApproachesOne(t *testing.T) {
	long := ""
	for i := 0; i < 1000; i++ {
		long += "x "
	}
	got, err := RepetitionRate(long, 3)
	if err != nil {
		t.Fatalf("RepetitionRate: %v", err)
	}
	if got <= 0.99 {
		t.Fatalf("rep = %v, want > 0.99 for a long degenerate text", got)
	}
}

func TestRepetitionRateValidation(t *testing.T) {
	if _, err := RepetitionRate("a b c", 0); err == nil {
		t.Fatal("want error for n=0")
	}
	if _, err := RepetitionRate("a b c", -1); err == nil {
		t.Fatal("want error for n<0")
	}
}

// Invariant (DESIGN §4.6): a degenerate (highly repetitive) sample is dropped;
// a clean sample is kept.
func TestQualityFilterDropsDegenerate(t *testing.T) {
	clean := Sample{Prompt: "what is the integral of x squared", Answer: "x cubed over three", Domain: "math"}
	// Answer loops the same trigram-rich phrase; rep is very high.
	degenerate := Sample{
		Prompt: "solve",
		Answer: "the answer is the answer is the answer is the answer is the answer is",
		Domain: "math",
	}
	out, err := QualityFilter([]Sample{clean, degenerate}, DefaultQualityParams())
	if err != nil {
		t.Fatalf("QualityFilter: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("kept %d samples, want 1", len(out))
	}
	if out[0].Prompt != clean.Prompt {
		t.Fatalf("kept the wrong sample: %q", out[0].Prompt)
	}
}

// Order is preserved and the threshold comparison is inclusive (≥).
func TestQualityFilterOrderAndThreshold(t *testing.T) {
	samples := []Sample{
		{Prompt: "a b c d e f", Domain: "x"},     // rep = 0
		{Prompt: "p q r s t u", Domain: "y"},     // rep = 0
		{Prompt: "z z z z z z z z", Domain: "z"}, // rep high
	}
	out, err := QualityFilter(samples, QualityParams{N: 2, MaxRepetition: 0.5})
	if err != nil {
		t.Fatalf("QualityFilter: %v", err)
	}
	if len(out) != 2 || out[0].Domain != "x" || out[1].Domain != "y" {
		t.Fatalf("filter did not preserve order/keep set: %+v", out)
	}

	// Threshold of exactly the sample's rate drops it (inclusive ≥).
	r, _ := RepetitionRate("z z z z z z z z", 2)
	exact := []Sample{{Prompt: "z z z z z z z z"}}
	dropped, _ := QualityFilter(exact, QualityParams{N: 2, MaxRepetition: r})
	if len(dropped) != 0 {
		t.Fatalf("threshold == rate should drop (inclusive), kept %d", len(dropped))
	}
}

func TestQualityFilterValidation(t *testing.T) {
	if _, err := QualityFilter(nil, QualityParams{N: 0, MaxRepetition: 0.5}); err == nil {
		t.Fatal("want error for n=0")
	}
	if _, err := QualityFilter(nil, QualityParams{N: 3, MaxRepetition: 1.5}); err == nil {
		t.Fatal("want error for threshold > 1")
	}
	if _, err := QualityFilter(nil, QualityParams{N: 3, MaxRepetition: -0.1}); err == nil {
		t.Fatal("want error for threshold < 0")
	}
}
