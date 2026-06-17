package instruct

import (
	"context"
	"testing"
)

func TestRulesIndividually(t *testing.T) {
	cases := []struct {
		name string
		rule Rule
		resp string
		want bool
	}{
		{"minwords-ok", MinWords{3}, "a b c d", true},
		{"minwords-fail", MinWords{3}, "a b", false},
		{"maxwords-ok", MaxWords{3}, "a b", true},
		{"maxwords-fail", MaxWords{3}, "a b c d", false},
		{"itemcount-ok", ItemCount{N: 2, Prefix: "-"}, "- one\n- two", true},
		{"itemcount-fail", ItemCount{N: 2, Prefix: "-"}, "- one\n- two\n- three", false},
		{"contain-ok", MustContain{[]string{"Apple", "PEAR"}}, "i like apple and pear", true},
		{"contain-fail", MustContain{[]string{"banana"}}, "apple pear", false},
		{"notcontain-ok", MustNotContain{[]string{"banana"}}, "apple pear", true},
		{"notcontain-fail", MustNotContain{[]string{"Pear"}}, "apple pear", false},
		{"ordering-ok", Ordering{[]string{"first", "second", "third"}}, "First then Second then THIRD", true},
		{"ordering-fail", Ordering{[]string{"first", "second"}}, "second comes before first", false},
		{"endswith-ok", EndsWith{"</answer>"}, "blah </answer>  \n", true},
		{"endswith-fail", EndsWith{"</answer>"}, "blah </answer> extra", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.rule.Check(c.resp); got != c.want {
				t.Fatalf("%s.Check(%q) = %v, want %v", c.rule.Describe(), c.resp, got, c.want)
			}
		})
	}
}

func TestRuleRewardBinary(t *testing.T) {
	rules := []Rule{MinWords{2}, MustContain{[]string{"hello"}}}
	if RuleReward("hello world", rules) != 1 {
		t.Fatal("satisfying response should reward 1")
	}
	if RuleReward("hi", rules) != 0 {
		t.Fatal("violating response should reward 0")
	}
	// Empty rule set is trivially satisfied.
	if RuleReward("anything", nil) != 1 {
		t.Fatal("no rules should reward 1")
	}
}

// fakeEnv is a stand-in rubric reward model.
type fakeEnv struct{ score float64 }

func (f fakeEnv) Score(ctx context.Context, prompt, completion string) (float64, error) {
	return f.score, nil
}

func TestComposerRoutesByKind(t *testing.T) {
	router := RouterFunc(func(prompt string) (PromptKind, []Rule) {
		if prompt == "open" {
			return OpenEnded, nil
		}
		return ExplicitConstraint, []Rule{MustContain{[]string{"ok"}}}
	})
	c, err := NewComposer(router, fakeEnv{score: 0.7})
	if err != nil {
		t.Fatalf("NewComposer: %v", err)
	}
	ctx := context.Background()

	// Explicit-constraint prompt scored 0/1 by its rules.
	if got, _ := c.Score(ctx, "constrained", "this is ok"); got != 1 {
		t.Fatalf("explicit-satisfied score = %v, want 1", got)
	}
	if got, _ := c.Score(ctx, "constrained", "nope"); got != 0 {
		t.Fatalf("explicit-violated score = %v, want 0", got)
	}
	// Open-ended prompt scored by the rubric.
	if got, _ := c.Score(ctx, "open", "whatever"); got != 0.7 {
		t.Fatalf("open-ended score = %v, want rubric 0.7", got)
	}
}

func TestComposerOpenEndedWithoutRubricFails(t *testing.T) {
	router := RouterFunc(func(string) (PromptKind, []Rule) { return OpenEnded, nil })
	c, err := NewComposer(router, nil) // no rubric
	if err != nil {
		t.Fatalf("NewComposer: %v", err)
	}
	if _, err := c.Score(context.Background(), "open", "x"); err == nil {
		t.Fatal("open-ended with nil rubric should error")
	}
}

func TestComposerNilRouter(t *testing.T) {
	if _, err := NewComposer(nil, nil); err == nil {
		t.Fatal("nil router should error")
	}
}
