package sandbox

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestFakeRunnerDrivesReward pins the §4.5 property: the fake runner's verdict
// drives a binary reward {0,1}.
func TestFakeRunnerDrivesReward(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name   string
		runner FakeRunner
		want   float64
	}{
		{"pass", FakeRunner{Pass: true}, 1},
		{"fail", FakeRunner{Pass: false}, 0},
		{"predicate-pass", FakeRunner{Predicate: func(code, tests string) bool {
			return strings.Contains(code, "return 42")
		}}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := reward(ctx, c.runner, "def f(): return 42", "assert f()==42")
			if err != nil {
				t.Fatalf("reward: %v", err)
			}
			if got != c.want {
				t.Fatalf("reward = %v, want %v", got, c.want)
			}
			if got != 0 && got != 1 {
				t.Fatalf("reward not binary: %v", got)
			}
		})
	}
}

func TestFakeRunnerError(t *testing.T) {
	sentinel := errors.New("compile failed")
	_, err := reward(context.Background(), FakeRunner{Err: sentinel}, "bad", "tests")
	if !errors.Is(err, sentinel) {
		t.Fatalf("runner error not propagated, got %v", err)
	}
}

// TestEnvironmentComposes checks the rl.RichEnvironment adapter scores binary
// and reports feedback on failure.
func TestEnvironmentComposes(t *testing.T) {
	ctx := context.Background()
	env, err := Environment(FakeRunner{Predicate: func(code, _ string) bool {
		return strings.Contains(code, "PASS")
	}}, "assert solve()")
	if err != nil {
		t.Fatalf("Environment: %v", err)
	}

	if got, _ := env.Score(ctx, "p", "PASS code"); got != 1 {
		t.Fatalf("passing code score = %v, want 1", got)
	}
	score, feedback, err := env.Verify(ctx, "p", "broken code")
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if score != 0 || feedback == "" {
		t.Fatalf("failing code = (%v,%q), want (0, non-empty feedback)", score, feedback)
	}
}

func TestEnvironmentNilRunner(t *testing.T) {
	if _, err := Environment(nil, "tests"); err == nil {
		t.Fatal("nil runner should error")
	}
}

// TestExecRunnerGated documents and enforces the gate: the real runner refuses
// to execute untrusted code by default, and even when allowed fails closed
// rather than running code without isolation.
func TestExecRunnerGated(t *testing.T) {
	ctx := context.Background()
	r := ExecRunner{Interpreter: "python3"}

	pass, err := r.Run(ctx, "print('hi')", "tests")
	if pass {
		t.Fatal("gated exec runner must not report pass")
	}
	if !errors.Is(err, ErrExecGated) {
		t.Fatalf("default exec runner error = %v, want ErrExecGated", err)
	}

	// Opting in alone still fails closed (no real sandbox wired in).
	pass, err = r.Allow().Run(ctx, "print('hi')", "tests")
	if pass {
		t.Fatal("allowed-but-unsandboxed exec runner must not report pass")
	}
	if err == nil || errors.Is(err, ErrExecGated) {
		t.Fatalf("allowed exec runner should fail closed with a distinct error, got %v", err)
	}
}

// TestExecRunnerFakeFallback shows the intended deployment pattern: when exec is
// gated, a trusted Runner is substituted to drive the reward.
func TestExecRunnerFakeFallback(t *testing.T) {
	var r Runner = ExecRunner{}
	if _, err := r.Run(context.Background(), "x", "y"); err == nil {
		t.Fatal("expected gated error before substitution")
	}
	r = FakeRunner{Pass: true} // trusted substitute
	pass, err := r.Run(context.Background(), "x", "y")
	if err != nil || !pass {
		t.Fatalf("substitute runner = (%v,%v), want (true,nil)", pass, err)
	}
}
