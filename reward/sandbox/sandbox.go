package sandbox

import (
	"context"
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// A Runner executes candidate code against a test program and reports whether
// the tests pass. It is the gate that isolates code execution from the rest of
// the reward pipeline: implementations range from an in-process fake (safe, for
// tests) to a real subprocess runner (gated, for deployments with isolation).
// Run must not panic on bad code; it returns (false, err) when execution itself
// fails (compile error, timeout, runner disabled) and (pass, nil) when the code
// ran and the tests reached a verdict.
type Runner interface {
	Run(ctx context.Context, code, tests string) (pass bool, err error)
}

// FakeRunner is an in-process Runner used by tests and toy pipelines. It never
// executes code: its verdict is the fixed Pass value, unless Predicate is set,
// in which case Predicate(code, tests) decides. It is safe by construction.
type FakeRunner struct {
	// Pass is the verdict returned when Predicate is nil.
	Pass bool
	// Predicate, when non-nil, computes the verdict from the inputs without
	// executing them (e.g. a substring check standing in for a test result).
	Predicate func(code, tests string) bool
	// Err, when non-nil, is returned as the execution error (verdict false),
	// modeling a runner-level failure.
	Err error
}

// Run implements Runner without executing anything.
func (f FakeRunner) Run(ctx context.Context, code, tests string) (bool, error) {
	if f.Err != nil {
		return false, f.Err
	}
	if f.Predicate != nil {
		return f.Predicate(code, tests), nil
	}
	return f.Pass, nil
}

// reward runs the code under r and maps the verdict to a binary reward. A
// runner error propagates (the caller decides whether to treat it as 0); a
// clean false verdict is reward 0, a true verdict is reward 1.
func reward(ctx context.Context, r Runner, code, tests string) (float64, error) {
	pass, err := r.Run(ctx, code, tests)
	if err != nil {
		return 0, err
	}
	if pass {
		return 1, nil
	}
	return 0, nil
}

// VerifyFor returns a VerifyFunc that treats the completion as the candidate
// code, runs it under r against the fixed tests, and yields a binary reward
// with diagnostic feedback. A runner error is surfaced as feedback with reward
// 0 (rather than as a hard error) so a single bad rollout does not abort a
// batch; use Runner.Run directly if you need the raw error.
func VerifyFor(r Runner, tests string) rl.VerifyFunc {
	return func(prompt, completion string) (float64, string, error) {
		pass, err := r.Run(context.Background(), completion, tests)
		if err != nil {
			return 0, "execution error: " + err.Error(), nil
		}
		if pass {
			return 1, "", nil
		}
		return 0, "tests failed", nil
	}
}

// Environment adapts r plus a fixed test program to an rl.RichEnvironment, so
// the sandbox reward composes with the MGPO/GRPO optimizer. The completion is
// the candidate code; the prompt is ignored.
func Environment(r Runner, tests string) (rl.RichEnvironment, error) {
	if r == nil {
		return nil, fmt.Errorf("sandbox: nil runner")
	}
	return rl.EnvFromVerifyFunc(VerifyFor(r, tests)), nil
}

var _ Runner = FakeRunner{}
