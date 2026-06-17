package instruct

import (
	"context"
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// PromptKind distinguishes the two reward routes of Instruct RL.
type PromptKind int

const (
	// ExplicitConstraint prompts are scored by rule-based validators.
	ExplicitConstraint PromptKind = iota
	// OpenEnded prompts are scored by the rubric reward model.
	OpenEnded
)

// A Router classifies a prompt and, for explicit-constraint prompts, supplies
// the rules its response must satisfy. It is how the caller wires a dataset's
// per-prompt constraints into the composed reward.
type Router interface {
	// Classify returns the prompt's kind and, when ExplicitConstraint, the
	// rules to check; rules are ignored for OpenEnded prompts.
	Classify(prompt string) (PromptKind, []Rule)
}

// RouterFunc adapts a function to Router.
type RouterFunc func(prompt string) (PromptKind, []Rule)

// Classify calls the wrapped function.
func (f RouterFunc) Classify(prompt string) (PromptKind, []Rule) { return f(prompt) }

// A Composer routes each prompt to its reward source and presents the result as
// a single rl.Environment for the MGPO/GRPO optimizer. Explicit-constraint
// prompts are scored 0/1 by their rules; open-ended prompts are scored by the
// injected rubric Environment (a gated reward model). The zero Composer is not
// usable; construct one with NewComposer.
type Composer struct {
	router Router
	rubric rl.Environment
}

// NewComposer builds a Composer. router is required. rubric is the open-ended
// reward source; it may be nil only if no prompt is ever classified OpenEnded
// (scoring an OpenEnded prompt with a nil rubric is an error).
func NewComposer(router Router, rubric rl.Environment) (*Composer, error) {
	if router == nil {
		return nil, fmt.Errorf("instruct: nil router")
	}
	return &Composer{router: router, rubric: rubric}, nil
}

// Score implements rl.Environment: it classifies the prompt and dispatches to
// the rule-based or rubric-based reward accordingly.
func (c *Composer) Score(ctx context.Context, prompt, completion string) (float64, error) {
	kind, rules := c.router.Classify(prompt)
	switch kind {
	case ExplicitConstraint:
		return RuleReward(completion, rules), nil
	case OpenEnded:
		if c.rubric == nil {
			return 0, fmt.Errorf("instruct: open-ended prompt but no rubric reward configured")
		}
		return c.rubric.Score(ctx, prompt, completion)
	default:
		return 0, fmt.Errorf("instruct: unknown prompt kind %d", kind)
	}
}

// Environment returns c as an rl.Environment (it already satisfies the
// interface); provided for readability at call sites.
func (c *Composer) Environment() rl.Environment { return c }
