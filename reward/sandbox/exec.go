package sandbox

import (
	"context"
	"errors"
)

// ErrExecGated is returned by a not-yet-allowed ExecRunner. Running untrusted
// model-generated code requires real isolation that this package does not
// provide, so the real runner is disabled by default and must be turned on
// deliberately.
var ErrExecGated = errors.New("sandbox: exec runner is gated; call Allow to enable (provide real isolation first)")

// ExecRunner is the real, subprocess-based Runner.
//
// GATE: executing model-generated code is unsafe without isolation, so this
// runner refuses to run until Allowed is set (via Allow). Even when allowed,
// this package performs NO sandboxing — it would shell out roughly as
//
//	<Interpreter> -c "<code>\n<tests>"   (e.g. Interpreter = "python3")
//
// in a child process. A real deployment MUST run that command inside a
// container / seccomp jail / network- and filesystem-restricted environment
// and enforce a timeout; the safe path is to keep ExecRunner gated and supply a
// trusted external Runner implementation instead. We therefore deliberately do
// NOT spawn a process here: Run returns ErrExecGated when not allowed, and when
// allowed returns errExecUnsandboxed so that turning the gate on without wiring
// real isolation still fails closed rather than executing untrusted code.
type ExecRunner struct {
	// Interpreter is the command that would run the code (e.g. "python3").
	Interpreter string
	// Allowed must be set to opt in to execution. Even then this runner does
	// not actually execute, by design — see the type doc.
	Allowed bool
}

// errExecUnsandboxed is returned when ExecRunner is allowed but no real sandbox
// is wired in. It keeps the gate fail-closed: enabling the flag alone never
// runs untrusted code.
var errExecUnsandboxed = errors.New("sandbox: exec runner allowed but no isolated sandbox is wired in; refusing to execute untrusted code")

// Allow returns a copy of r with execution opted in. It exists to make the gate
// explicit and greppable at call sites.
func (r ExecRunner) Allow() ExecRunner {
	r.Allowed = true
	return r
}

// Run implements Runner. It is gated: it never executes untrusted code in this
// package. When not allowed it returns ErrExecGated; when allowed it returns
// errExecUnsandboxed, documenting that real isolation must be supplied by an
// external Runner before any execution happens.
func (r ExecRunner) Run(ctx context.Context, code, tests string) (bool, error) {
	if !r.Allowed {
		return false, ErrExecGated
	}
	return false, errExecUnsandboxed
}

var _ Runner = ExecRunner{}
