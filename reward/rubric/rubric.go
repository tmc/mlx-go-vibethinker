package rubric

import (
	"context"
	"errors"
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// A Scorer grades a completion against a textual rubric, returning a scalar
// quality score. It is the reward-model gate: the real implementation is a
// frontier judge model supplied by the caller; FakeScorer stands in for tests.
// Score may return any real value; the Environment adapter clamps it to [0,1].
type Scorer interface {
	Score(ctx context.Context, rubric, prompt, completion string) (float64, error)
}

// ScorerFunc adapts a function to Scorer.
type ScorerFunc func(ctx context.Context, rubric, prompt, completion string) (float64, error)

// Score calls the wrapped function.
func (f ScorerFunc) Score(ctx context.Context, rubric, prompt, completion string) (float64, error) {
	return f(ctx, rubric, prompt, completion)
}

// FakeScorer is a deterministic in-repo Scorer for tests and toy pipelines. It
// returns Value, unless Fn is set, in which case Fn decides. It performs no
// model call and no I/O.
type FakeScorer struct {
	// Value is the score returned when Fn is nil.
	Value float64
	// Fn, when non-nil, computes the score from the inputs.
	Fn func(rubric, prompt, completion string) float64
	// Err, when non-nil, is returned as the scoring error.
	Err error
}

// Score implements Scorer deterministically.
func (s FakeScorer) Score(ctx context.Context, rubric, prompt, completion string) (float64, error) {
	if s.Err != nil {
		return 0, s.Err
	}
	if s.Fn != nil {
		return s.Fn(rubric, prompt, completion), nil
	}
	return s.Value, nil
}

// ErrScorerGated is returned by GatedScorer. The real rubric reward model is a
// frontier judge that cannot run locally; it must be supplied explicitly.
var ErrScorerGated = errors.New("rubric: reward model is gated; supply a concrete Scorer")

// GatedScorer is a placeholder Scorer that names the reward model a real run
// would call and fails closed until replaced. It documents the gate without
// bundling a model.
type GatedScorer struct {
	// Model is the identifier of the reward model that would be called
	// (e.g. an API model name); informational only.
	Model string
}

// Score always fails closed with ErrScorerGated.
func (g GatedScorer) Score(ctx context.Context, rubric, prompt, completion string) (float64, error) {
	return 0, fmt.Errorf("rubric: model %q: %w", g.Model, ErrScorerGated)
}

// A Rubric source supplies the grading rubric for a prompt. It is how a
// dataset's per-prompt grading criteria are wired into the reward.
type Rubric interface {
	For(prompt string) string
}

// RubricFunc adapts a function to Rubric.
type RubricFunc func(prompt string) string

// For calls the wrapped function.
func (f RubricFunc) For(prompt string) string { return f(prompt) }

// StaticRubric is a single rubric used for every prompt.
type StaticRubric string

// For returns the static rubric text.
func (r StaticRubric) For(string) string { return string(r) }

// clamp01 maps any real (including NaN) into [0,1]: NaN and values below 0 go
// to 0, values above 1 go to 1. This keeps the rubric reward in the binary
// reward's [0,1] envelope the optimizer expects.
func clamp01(v float64) float64 {
	if !(v > 0) { // false for NaN and v<=0
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// env adapts a Scorer and Rubric source to rl.Environment.
type env struct {
	scorer Scorer
	rubric Rubric
}

// Score implements rl.Environment: it looks up the prompt's rubric, scores the
// completion, and clamps the result into [0,1].
func (e *env) Score(ctx context.Context, prompt, completion string) (float64, error) {
	r := e.rubric.For(prompt)
	v, err := e.scorer.Score(ctx, r, prompt, completion)
	if err != nil {
		return 0, fmt.Errorf("rubric: score: %w", err)
	}
	return clamp01(v), nil
}

// Environment adapts a Scorer and per-prompt Rubric source to an
// rl.Environment, so the rubric reward composes with the MGPO/GRPO optimizer.
// Both arguments are required.
func Environment(scorer Scorer, rubric Rubric) (rl.Environment, error) {
	if scorer == nil {
		return nil, fmt.Errorf("rubric: nil scorer")
	}
	if rubric == nil {
		return nil, fmt.Errorf("rubric: nil rubric source")
	}
	return &env{scorer: scorer, rubric: rubric}, nil
}

var (
	_ Scorer         = FakeScorer{}
	_ Scorer         = GatedScorer{}
	_ rl.Environment = (*env)(nil)
)
