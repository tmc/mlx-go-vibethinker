package data

import (
	"fmt"
	"strings"
)

// QualityParams configures the n-gram repetition/degeneration filter (DESIGN
// §4.6). N is the n-gram order; MaxRepetition is the inclusive upper bound on a
// sample's repetition rate — a sample whose rep_N ≥ MaxRepetition is dropped.
type QualityParams struct {
	// N is the n-gram order used to measure repetition (e.g. 3 for trigrams).
	N int

	// MaxRepetition is the repetition-rate threshold in [0,1]. A sample with
	// rep_N at or above this value is dropped as degenerate. A value of 1 keeps
	// everything (only a fully-degenerate rep_N = 1 could be dropped, and the
	// comparison is ≥, so 1 effectively never drops).
	MaxRepetition float64
}

// DefaultQualityParams returns trigram filtering with a 0.7 repetition cap:
// drop a sample where 70% or more of its trigrams are repeats.
func DefaultQualityParams() QualityParams {
	return QualityParams{N: 3, MaxRepetition: 0.7}
}

// RepetitionRate returns the fraction of a text's order-n token n-grams that are
// repeats:
//
//	rep_n = 1 − distinct_n / total_n.
//
// Text is split on whitespace into tokens. A text with fewer than n tokens has
// no n-grams; its rate is 0 (too short to be degenerate by repetition). All
// distinct n-grams ⇒ 0; a text that is one token repeated has every n-gram
// identical ⇒ rep_n = (total−1)/total → 1 as the text grows. n must be
// positive.
func RepetitionRate(text string, n int) (float64, error) {
	if n <= 0 {
		return 0, fmt.Errorf("data: n must be positive, got %d", n)
	}
	return repetitionRate(strings.Fields(text), n), nil
}

// repetitionRate is the unchecked core of RepetitionRate over pre-split tokens.
func repetitionRate(tokens []string, n int) float64 {
	total := len(tokens) - n + 1
	if total <= 0 {
		return 0 // fewer than n tokens: no n-grams
	}
	seen := make(map[string]struct{}, total)
	for i := 0; i < total; i++ {
		seen[strings.Join(tokens[i:i+n], " ")] = struct{}{}
	}
	return 1 - float64(len(seen))/float64(total)
}

// QualityFilter drops samples whose n-gram repetition rate (over Prompt joined
// with Answer) is at or above MaxRepetition, returning the kept samples in input
// order. It is a thin shell over the repetition core. N must be positive and
// MaxRepetition must be in [0,1].
func QualityFilter(samples []Sample, p QualityParams) ([]Sample, error) {
	if p.N <= 0 {
		return nil, fmt.Errorf("data: n must be positive, got %d", p.N)
	}
	if p.MaxRepetition < 0 || p.MaxRepetition > 1 {
		return nil, fmt.Errorf("data: max repetition must be in [0,1], got %v", p.MaxRepetition)
	}
	return filter(samples, p), nil
}

// filter is the unchecked core of QualityFilter.
func filter(samples []Sample, p QualityParams) []Sample {
	out := make([]Sample, 0, len(samples))
	for _, s := range samples {
		text := s.Prompt
		if s.Answer != "" {
			text = s.Prompt + " " + s.Answer
		}
		if repetitionRate(strings.Fields(text), p.N) >= p.MaxRepetition {
			continue
		}
		out = append(out, s)
	}
	return out
}
