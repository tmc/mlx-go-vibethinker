package rubric

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

// TestFakeScorerDrivesReward pins the §4.5 property: the fake scorer drives the
// rubric reward through the Environment adapter.
func TestFakeScorerDrivesReward(t *testing.T) {
	ctx := context.Background()
	env, err := Environment(FakeScorer{Value: 0.75}, StaticRubric("be concise"))
	if err != nil {
		t.Fatalf("Environment: %v", err)
	}
	got, err := env.Score(ctx, "explain X", "a concise explanation")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got != 0.75 {
		t.Fatalf("score = %v, want 0.75", got)
	}
}

// TestScorerSeesRubricAndCompletion checks the per-prompt rubric and completion
// reach the scorer, so grading is rubric-conditioned.
func TestScorerSeesRubricAndCompletion(t *testing.T) {
	scorer := FakeScorer{Fn: func(rubric, prompt, completion string) float64 {
		if strings.Contains(completion, rubric) {
			return 1
		}
		return 0
	}}
	rubricSrc := RubricFunc(func(prompt string) string {
		if prompt == "p1" {
			return "alpha"
		}
		return "beta"
	})
	env, err := Environment(scorer, rubricSrc)
	if err != nil {
		t.Fatalf("Environment: %v", err)
	}
	ctx := context.Background()
	if got, _ := env.Score(ctx, "p1", "contains alpha here"); got != 1 {
		t.Fatalf("rubric-matching completion = %v, want 1", got)
	}
	if got, _ := env.Score(ctx, "p1", "contains beta here"); got != 0 {
		t.Fatalf("non-matching completion = %v, want 0", got)
	}
}

// TestClampToUnitInterval pins the [0,1] envelope.
func TestClampToUnitInterval(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{-3, 0},
		{0, 0},
		{0.5, 0.5},
		{1, 1},
		{2.5, 1},
		{math.NaN(), 0},
	}
	for _, c := range cases {
		env, err := Environment(FakeScorer{Value: c.in}, StaticRubric("r"))
		if err != nil {
			t.Fatalf("Environment: %v", err)
		}
		got, err := env.Score(context.Background(), "p", "y")
		if err != nil {
			t.Fatalf("Score: %v", err)
		}
		if got != c.want {
			t.Fatalf("clamp(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestScorerErrorPropagates(t *testing.T) {
	sentinel := errors.New("model unavailable")
	env, err := Environment(FakeScorer{Err: sentinel}, StaticRubric("r"))
	if err != nil {
		t.Fatalf("Environment: %v", err)
	}
	if _, err := env.Score(context.Background(), "p", "y"); !errors.Is(err, sentinel) {
		t.Fatalf("scorer error not propagated, got %v", err)
	}
}

// TestGatedScorerFailsClosed documents the gate: the real reward model is not
// bundled, so the gated scorer refuses to score.
func TestGatedScorerFailsClosed(t *testing.T) {
	env, err := Environment(GatedScorer{Model: "judge-xl"}, StaticRubric("r"))
	if err != nil {
		t.Fatalf("Environment: %v", err)
	}
	_, err = env.Score(context.Background(), "p", "y")
	if !errors.Is(err, ErrScorerGated) {
		t.Fatalf("gated scorer error = %v, want ErrScorerGated", err)
	}
}

func TestEnvironmentNilArgs(t *testing.T) {
	if _, err := Environment(nil, StaticRubric("r")); err == nil {
		t.Fatal("nil scorer should error")
	}
	if _, err := Environment(FakeScorer{}, nil); err == nil {
		t.Fatal("nil rubric should error")
	}
}
