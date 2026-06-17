package multidomain

import (
	"context"
	"errors"
	"testing"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

func env(score float64) rl.Environment {
	return rl.EnvFromFunc(func(prompt, completion string) (float64, error) { return score, nil })
}

// fakeRunner records the domain order it saw and the input checkpoint per
// domain, and emits a deterministic output dir.
type fakeRunner struct {
	order   []string
	inputs  []string
	failOn  string
	failErr error
}

func (f *fakeRunner) RunDomain(ctx context.Context, d Domain, inDir string) (string, error) {
	f.order = append(f.order, d.Name)
	f.inputs = append(f.inputs, inDir)
	if d.Name == f.failOn {
		return "", f.failErr
	}
	return inDir + "/" + d.Name, nil
}

func TestRunOrdersDomainsAndThreadsCheckpoint(t *testing.T) {
	domains := DefaultOrder(env(1), env(1), env(1))
	r := &fakeRunner{}
	results, final, err := Run(context.Background(), r, "/ck/sft", domains)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Order Math -> Code -> STEM.
	wantOrder := []string{"math", "code", "stem"}
	for i, w := range wantOrder {
		if r.order[i] != w {
			t.Fatalf("domain %d = %q, want %q", i, r.order[i], w)
		}
	}
	// Checkpoint threaded: each domain's input is the previous output.
	wantInputs := []string{"/ck/sft", "/ck/sft/math", "/ck/sft/math/code"}
	for i, w := range wantInputs {
		if r.inputs[i] != w {
			t.Fatalf("domain %d input = %q, want %q", i, r.inputs[i], w)
		}
	}
	if final != "/ck/sft/math/code/stem" {
		t.Fatalf("final = %q", final)
	}
	// Every intermediate checkpoint retained, in order.
	if len(results) != 3 {
		t.Fatalf("retained %d results, want 3", len(results))
	}
	if results[0].Domain != "math" || results[0].Dir != "/ck/sft/math" {
		t.Fatalf("results[0] = %+v", results[0])
	}
	if results[2].Dir != final {
		t.Fatalf("last retained dir %q != final %q", results[2].Dir, final)
	}
}

func TestRunStopsOnError(t *testing.T) {
	sentinel := errors.New("rl boom")
	domains := DefaultOrder(env(1), env(1), env(1))
	r := &fakeRunner{failOn: "code", failErr: sentinel}
	_, _, err := Run(context.Background(), r, "/ck/sft", domains)
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want wrap of sentinel", err)
	}
	// STEM must not have run after code failed.
	for _, d := range r.order {
		if d == "stem" {
			t.Fatal("stem ran after code failed")
		}
	}
}

func TestRunRejectsNilRewardAndRunner(t *testing.T) {
	if _, _, err := Run(context.Background(), nil, "/x", nil); err == nil {
		t.Fatal("nil runner should error")
	}
	domains := []Domain{{Name: "math", Reward: nil}}
	if _, _, err := Run(context.Background(), &fakeRunner{}, "/x", domains); err == nil {
		t.Fatal("nil reward should error")
	}
}
