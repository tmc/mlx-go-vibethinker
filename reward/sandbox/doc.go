// Package sandbox implements VibeThinker's code-execution reward (DESIGN §4.5:
// "sandbox — execute generated code against tests; Environment via
// EnvFromVerifyFunc. Local runner with a safe-exec seam").
//
// The reward is binary: a completion scores 1 when its code passes the
// supplied tests under a Runner, and 0 otherwise —
//
//	r(q, y) = 1 if Run(code(y), tests) passes, else 0.
//
// Executing model-generated code is the genuinely dangerous step, so it sits
// behind a Runner interface (a clearly-documented gate) rather than being
// performed unconditionally:
//
//   - FakeRunner is an in-process, side-effect-free Runner driven by a fixed
//     pass/fail decision (optionally a per-code predicate). It is SAFE by
//     construction and is what the tests and toy pipelines use.
//   - ExecRunner is the real os/exec-based runner. It is GATED: it returns an
//     error until explicitly enabled with Allow, because it runs untrusted code
//     in a child process. A production deployment must supply real isolation
//     (container, seccomp, network/file restrictions); this package does not,
//     and says so. The exact command it would run and why it is not run by
//     default are documented on ExecRunner.
//
// Verify and Environment adapt a Runner plus a fixed test program to mlx-go-rl's
// RichEnvironment, so the sandbox reward composes with the MGPO/GRPO optimizer.
package sandbox
