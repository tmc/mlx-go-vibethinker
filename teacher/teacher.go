package teacher

import (
	"context"
	"fmt"
	"hash/fnv"
)

// A Trace is one reasoning path generated for a prompt, with the final answer
// the teacher extracted from it.
type Trace struct {
	Text   string
	Answer string
}

// A Claim is a decision-relevant statement extracted from a trace, used by the
// 3B's Claim-Level Reliability scaling. Valid is the teacher's self-verification
// verdict.
type Claim struct {
	Text  string
	Valid bool
}

// Teacher is the strong-model gate. A real implementation calls a frontier model;
// the in-repo Fake is deterministic. All methods take a context so a real
// implementation can honor cancellation and deadlines.
type Teacher interface {
	// Traces returns n multi-path reasoning traces for the prompt.
	Traces(ctx context.Context, prompt string, n int) ([]Trace, error)

	// PseudoLabel returns a majority-vote answer for the prompt across sampled
	// traces, plus the vote fraction in [0,1].
	PseudoLabel(ctx context.Context, prompt string, n int) (answer string, confidence float64, err error)

	// Claims extracts up to m decision-relevant claims from a trace and
	// self-verifies each.
	Claims(ctx context.Context, trace Trace, m int) ([]Claim, error)
}

// Fake is a deterministic in-repo Teacher for tests and the toy pipeline. It
// derives traces and answers from a hash of the prompt so runs are reproducible
// without any model. It is not a substitute for a real teacher in a real run.
type Fake struct {
	// AnswerFor optionally maps a prompt to a fixed answer; when nil or absent
	// the answer is derived from the prompt hash.
	AnswerFor map[string]string
}

// Traces returns n deterministic traces for the prompt.
func (f Fake) Traces(ctx context.Context, prompt string, n int) ([]Trace, error) {
	if n <= 0 {
		return nil, fmt.Errorf("teacher: n must be positive, got %d", n)
	}
	ans := f.answer(prompt)
	out := make([]Trace, n)
	for i := range out {
		out[i] = Trace{
			Text:   fmt.Sprintf("reasoning(%s)#%d => %s", prompt, i, ans),
			Answer: ans,
		}
	}
	return out, nil
}

// PseudoLabel returns the deterministic answer with full confidence (the fake's
// traces all agree).
func (f Fake) PseudoLabel(ctx context.Context, prompt string, n int) (string, float64, error) {
	if n <= 0 {
		return "", 0, fmt.Errorf("teacher: n must be positive, got %d", n)
	}
	return f.answer(prompt), 1.0, nil
}

// Claims returns m claims derived from the trace; the fake marks all valid.
func (f Fake) Claims(ctx context.Context, trace Trace, m int) ([]Claim, error) {
	if m <= 0 {
		return nil, fmt.Errorf("teacher: m must be positive, got %d", m)
	}
	out := make([]Claim, m)
	for i := range out {
		out[i] = Claim{Text: fmt.Sprintf("claim#%d of %q", i, trace.Answer), Valid: true}
	}
	return out, nil
}

func (f Fake) answer(prompt string) string {
	if a, ok := f.AnswerFor[prompt]; ok {
		return a
	}
	h := fnv.New32a()
	h.Write([]byte(prompt))
	return fmt.Sprintf("%d", h.Sum32()%100)
}
