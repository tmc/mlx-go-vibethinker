//go:build modelir

package recipe

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/mlx-go-vibethinker/internal/toymodel"
	"github.com/tmc/mlx-go-vibethinker/ssp"
)

func newToy(t *testing.T) *Toy {
	t.Helper()
	lm, err := toymodel.New(toymodel.DefaultConfig(), 1)
	if err != nil {
		t.Fatalf("toymodel.New: %v", err)
	}
	return &Toy{Model: lm, Tok: toymodel.Tokenizer{}, Dir: t.TempDir()}
}

// runPipeline runs p and returns the final checkpoint, failing on any error
// (stages return an error on a non-finite loss, so success means no NaNs).
func runPipeline(t *testing.T, p *ssp.Pipeline) *ssp.Checkpoint {
	t.Helper()
	var seen []string
	p.Observe = func(ev ssp.Event) {
		if ev.Kind == ssp.StageDone {
			seen = append(seen, ev.Stage)
		}
	}
	out, err := p.Run(context.Background(), &ssp.Checkpoint{})
	if err != nil {
		t.Fatalf("pipeline %q failed (NaN or stage error): %v", p.Name, err)
	}
	if out == nil {
		t.Fatalf("pipeline %q returned nil checkpoint", p.Name)
	}
	if len(seen) != len(p.Stages) {
		t.Fatalf("pipeline %q ran %d/%d stages", p.Name, len(seen), len(p.Stages))
	}
	return out
}

// The 1.5B toy recipe runs end to end and emits a merged + RL-updated
// checkpoint with provenance and no NaNs.
func TestPipeline15BEndToEnd(t *testing.T) {
	toy := newToy(t)
	out := runPipeline(t, toy.Pipeline15B())

	// Final checkpoint file exists.
	if _, err := os.Stat(out.Dir); err != nil {
		t.Fatalf("final checkpoint %q missing: %v", out.Dir, err)
	}
	// Provenance recorded for every stage, in order.
	wantStages := []string{"sft", "probe", "fuse", "mgpo-math", "mgpo-code"}
	if len(out.History) != len(wantStages) {
		t.Fatalf("history has %d entries, want %d: %+v", len(out.History), len(wantStages), out.History)
	}
	for i, w := range wantStages {
		if out.History[i].Stage != w {
			t.Fatalf("history[%d] = %q, want %q", i, out.History[i].Stage, w)
		}
	}
	// A fusion stage and an MGPO stage both contributed.
	assertProvenanceContains(t, out, "fuse", "fused")
	assertProvenanceContains(t, out, "mgpo-math", "mgpo loss")
}

// The 3B toy recipe runs the full consolidation flow end to end, ending in
// offline self-distillation, with no NaNs.
func TestPipeline3BEndToEnd(t *testing.T) {
	toy := newToy(t)
	out := runPipeline(t, toy.Pipeline3B())

	if _, err := os.Stat(out.Dir); err != nil {
		t.Fatalf("final checkpoint %q missing: %v", out.Dir, err)
	}
	// The last stage is offline self-distillation.
	last := out.History[len(out.History)-1]
	if last.Stage != "distill" {
		t.Fatalf("final stage = %q, want distill", last.Stage)
	}
	// Both curriculum SFT stages and all three MGPO domains ran.
	for _, name := range []string{"sft-broad", "sft-hard", "mgpo-math", "mgpo-code", "mgpo-stem", "distill"} {
		if !hasStage(out, name) {
			t.Fatalf("3B history missing stage %q: %+v", name, stageNames(out))
		}
	}
}

func assertProvenanceContains(t *testing.T, c *ssp.Checkpoint, stage, noteSub string) {
	t.Helper()
	for _, p := range c.History {
		if p.Stage == stage {
			if noteSub != "" && !contains(p.Note, noteSub) {
				t.Fatalf("stage %q note %q does not contain %q", stage, p.Note, noteSub)
			}
			return
		}
	}
	t.Fatalf("provenance missing stage %q", stage)
}

func hasStage(c *ssp.Checkpoint, stage string) bool {
	for _, p := range c.History {
		if p.Stage == stage {
			return true
		}
	}
	return false
}

func stageNames(c *ssp.Checkpoint) []string {
	var out []string
	for _, p := range c.History {
		out = append(out, p.Stage)
	}
	return out
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
