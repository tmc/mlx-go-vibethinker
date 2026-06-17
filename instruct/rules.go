package instruct

import (
	"strings"
	"unicode"
)

// A Rule is a verifiable constraint on a response. Check reports whether the
// response satisfies the constraint. Rules are pure and deterministic so they
// give a reliable binary reward signal.
type Rule interface {
	Check(response string) bool
	Describe() string
}

// AllRules reports whether a response satisfies every rule. An empty rule set is
// trivially satisfied.
func AllRules(response string, rules []Rule) bool {
	for _, r := range rules {
		if !r.Check(response) {
			return false
		}
	}
	return true
}

// RuleReward returns 1 when the response satisfies all rules and 0 otherwise —
// the binary reward Instruct RL uses for explicit-constraint prompts.
func RuleReward(response string, rules []Rule) float64 {
	if AllRules(response, rules) {
		return 1
	}
	return 0
}

// --- Concrete rules ---

// MinWords requires at least N whitespace-separated words.
type MinWords struct{ N int }

func (r MinWords) Check(s string) bool { return len(strings.Fields(s)) >= r.N }
func (r MinWords) Describe() string    { return "minimum word count" }

// MaxWords requires at most N whitespace-separated words.
type MaxWords struct{ N int }

func (r MaxWords) Check(s string) bool { return len(strings.Fields(s)) <= r.N }
func (r MaxWords) Describe() string    { return "maximum word count" }

// ItemCount requires exactly N lines that begin with the given bullet prefix
// (after trimming leading whitespace), e.g. prefix "-" or "1." style markers.
type ItemCount struct {
	N      int
	Prefix string
}

func (r ItemCount) Check(s string) bool {
	count := 0
	for line := range strings.SplitSeq(s, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), r.Prefix) {
			count++
		}
	}
	return count == r.N
}
func (r ItemCount) Describe() string { return "exact item count" }

// MustContain requires the response to contain every keyword (case-insensitive).
type MustContain struct{ Keywords []string }

func (r MustContain) Check(s string) bool {
	low := strings.ToLower(s)
	for _, k := range r.Keywords {
		if !strings.Contains(low, strings.ToLower(k)) {
			return false
		}
	}
	return true
}
func (r MustContain) Describe() string { return "required keywords present" }

// MustNotContain requires the response to contain none of the keywords
// (case-insensitive).
type MustNotContain struct{ Keywords []string }

func (r MustNotContain) Check(s string) bool {
	low := strings.ToLower(s)
	for _, k := range r.Keywords {
		if strings.Contains(low, strings.ToLower(k)) {
			return false
		}
	}
	return true
}
func (r MustNotContain) Describe() string { return "forbidden keywords absent" }

// Ordering requires the given substrings to appear in the response in the given
// order (case-insensitive), each after the previous one's match.
type Ordering struct{ Sequence []string }

func (r Ordering) Check(s string) bool {
	low := strings.ToLower(s)
	pos := 0
	for _, sub := range r.Sequence {
		i := strings.Index(low[pos:], strings.ToLower(sub))
		if i < 0 {
			return false
		}
		pos += i + len(sub)
	}
	return true
}
func (r Ordering) Describe() string { return "required ordering" }

// EndsWith requires the trimmed response to end with the given marker — a
// completion check (e.g. a closing token or sentinel).
type EndsWith struct{ Marker string }

func (r EndsWith) Check(s string) bool {
	return strings.HasSuffix(strings.TrimRightFunc(s, unicode.IsSpace), r.Marker)
}
func (r EndsWith) Describe() string { return "completion marker" }
