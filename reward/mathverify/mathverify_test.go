package mathverify

import (
	"context"
	"testing"
)

func TestExtractAnswer(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
		ok   bool
	}{
		{"boxed", `The result is \boxed{42}.`, "42", true},
		{"last-boxed", `\boxed{1} then \boxed{2}`, "2", true},
		{"boxed-nested-braces", `\boxed{\frac{1}{2}}`, `\frac{1}{2}`, true},
		{"boxed-fbox", `\fbox{7}`, "7", true},
		{"trailing-number", "Reasoning here.\nThe answer is 3.14", "3.14", true},
		{"trailing-fraction", "so x = 2/4", "2/4", true},
		{"thousands", "total: 1,000", "1,000", true},
		{"symbolic-line", "the answer is\nx+y", "x+y", true},
		{"empty", "   ", "", false},
		{"empty-boxed-falls-through", `\boxed{}  and then 5`, "5", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := ExtractAnswer(c.in)
			if ok != c.ok || got != c.want {
				t.Fatalf("ExtractAnswer(%q) = (%q,%v), want (%q,%v)", c.in, got, ok, c.want, c.ok)
			}
		})
	}
}

func TestEquivalent(t *testing.T) {
	cases := []struct {
		name string
		a, b string
		want bool
	}{
		{"fraction-decimal", "2/4", "0.5", true},
		{"trailing-zero", "12", "12.0", true},
		{"trailing-zeros", "0.50", "0.5", true},
		{"thousands-comma", "1,000", "1000", true},
		{"latex-thousands", `1{,}000`, "1000", true},
		{"dollar-wrap", "$7$", "7", true},
		{"text-wrap", `\text{15}`, "15", true},
		{"negative", "-3", "-3.00", true},
		{"non-equivalent-num", "12", "13", false},
		{"non-equivalent-frac", "1/3", "0.5", false},
		{"symbolic-equal", "x+y", "x+y", true},
		{"symbolic-vs-number", "x", "5", false},
		{"empty-left", "", "5", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Equivalent(c.a, c.b); got != c.want {
				t.Fatalf("Equivalent(%q,%q) = %v, want %v", c.a, c.b, got, c.want)
			}
		})
	}
}

// TestRewardBinary pins the §4.5 property: the reward is binary, equivalence
// scores 1, non-equivalence and missing answers score 0.
func TestRewardBinary(t *testing.T) {
	cases := []struct {
		name       string
		completion string
		reference  string
		want       float64
	}{
		{"boxed-equivalent", `work...\boxed{0.5}`, `\boxed{2/4}`, 1},
		{"trailing-equal", "answer: 12.0", "12", 1},
		{"non-equivalent", `\boxed{7}`, `\boxed{8}`, 0},
		{"missing-completion-answer", "   ", "5", 0},
		{"missing-reference-answer", `\boxed{5}`, "", 0},
		{"thousands-equiv", "1,000", "1000", 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Reward(c.completion, c.reference); got != c.want {
				t.Fatalf("Reward(%q,%q) = %v, want %v", c.completion, c.reference, got, c.want)
			}
			if got := Reward(c.completion, c.reference); got != 0 && got != 1 {
				t.Fatalf("reward not binary: %v", got)
			}
		})
	}
}

// TestEnvironmentDrivesReward checks the rl.RichEnvironment adapter yields the
// same binary reward and feedback semantics.
func TestEnvironmentDrivesReward(t *testing.T) {
	env := Environment(`\boxed{2/4}`)
	ctx := context.Background()

	score, err := env.Score(ctx, "compute 1/2", `final \boxed{0.5}`)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 1 {
		t.Fatalf("equivalent completion score = %v, want 1", score)
	}

	score, feedback, err := env.Verify(ctx, "compute 1/2", `final \boxed{0.6}`)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if score != 0 {
		t.Fatalf("non-equivalent score = %v, want 0", score)
	}
	if feedback == "" {
		t.Fatal("failed completion should carry feedback")
	}

	// Missing answer scores 0 with explanatory feedback.
	score, feedback, err = env.Verify(ctx, "compute 1/2", "I am not sure.")
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if score != 0 || feedback == "" {
		t.Fatalf("missing answer = (%v,%q), want (0, non-empty)", score, feedback)
	}
}

func TestNormalizeIdempotent(t *testing.T) {
	// Normalizing an already-normalized string is a fixed point.
	for _, s := range []string{"1000", "0.5", "x+y", "-3", `\frac{1}{2}`} {
		n := Normalize(s)
		if Normalize(n) != n {
			t.Fatalf("Normalize not idempotent on %q: %q -> %q", s, n, Normalize(n))
		}
	}
}
