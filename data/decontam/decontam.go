package decontam

import (
	"fmt"
	"strings"
	"unicode"
)

// DefaultN is the paper's contamination n-gram order (10-gram overlap).
const DefaultN = 10

// Normalize lowercases text, drops every rune that is not a letter, digit, or
// space (stripping punctuation and symbols), and collapses runs of whitespace
// to single spaces, trimming the ends. The result is the canonical form whose
// n-grams are compared. Letters and digits keep Unicode semantics, so case
// folding and digit recognition work beyond ASCII.
func Normalize(text string) string {
	return strings.Join(tokenize(text), " ")
}

// tokenize is the shared normalization+split core: it returns the normalized
// tokens of text. A token is a maximal run of letters/digits; everything else
// is a separator. All tokens are lowercased.
func tokenize(text string) []string {
	var tokens []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			tokens = append(tokens, b.String())
			b.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(unicode.ToLower(r))
		default:
			flush()
		}
	}
	flush()
	return tokens
}

// NGrams returns the set of contiguous order-n token n-grams of text after
// normalization. A text with fewer than n tokens has no n-grams and yields an
// empty set (so it can never collide). n must be positive. Each n-gram is the
// space-joined normalized tokens; the map value is unused.
func NGrams(text string, n int) (map[string]struct{}, error) {
	if n <= 0 {
		return nil, fmt.Errorf("n must be positive, got %d", n)
	}
	return ngrams(tokenize(text), n), nil
}

// ngrams is the unchecked core of NGrams over pre-tokenized text.
func ngrams(tokens []string, n int) map[string]struct{} {
	set := make(map[string]struct{})
	for i := 0; i+n <= len(tokens); i++ {
		set[strings.Join(tokens[i:i+n], " ")] = struct{}{}
	}
	return set
}

// Filter returns the train texts that are NOT contaminated by any eval text: a
// train text is dropped iff it shares at least one order-n n-gram with the union
// of the eval texts' n-grams (DESIGN §4.6). Kept texts are returned in input
// order. n must be positive; pass [DefaultN] for the paper's 10-gram rule.
func Filter(train, eval []string, n int) ([]string, error) {
	if n <= 0 {
		return nil, fmt.Errorf("n must be positive, got %d", n)
	}
	return filter(train, eval, n), nil
}

// filter is the unchecked core of Filter.
func filter(train, eval []string, n int) []string {
	// Build the union eval n-gram set once.
	evalSet := make(map[string]struct{})
	for _, e := range eval {
		for g := range ngrams(tokenize(e), n) {
			evalSet[g] = struct{}{}
		}
	}
	out := make([]string, 0, len(train))
	for _, t := range train {
		if !overlaps(ngrams(tokenize(t), n), evalSet) {
			out = append(out, t)
		}
	}
	return out
}

// overlaps reports whether a and b share any key. It scans the smaller map for
// fewer lookups.
func overlaps(a, b map[string]struct{}) bool {
	if len(b) < len(a) {
		a, b = b, a
	}
	for g := range a {
		if _, ok := b[g]; ok {
			return true
		}
	}
	return false
}
