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
	// The terminal stage is Instruct RL (DESIGN §4: … self-distill -> Instruct RL).
	last := out.History[len(out.History)-1]
	if last.Stage != "instruct-rl" {
		t.Fatalf("final stage = %q, want instruct-rl", last.Stage)
	}
	// Both curriculum SFT stages, all three MGPO domains, Long2Short math,
	// distillation, and Instruct RL ran — the full DESIGN §4 ordering.
	for _, name := range []string{"sft-broad", "sft-hard", "mgpo-math", "long2short", "mgpo-code", "mgpo-stem", "distill", "instruct-rl"} {
		if !hasStage(out, name) {
			t.Fatalf("3B history missing stage %q: %+v", name, stageNames(out))
		}
	}
	// Long2Short must sit between MGPO-math and MGPO-code (DESIGN §4 ordering).
	if i, j, k := stageIndex(out, "mgpo-math"), stageIndex(out, "long2short"), stageIndex(out, "mgpo-code"); !(i < j && j < k) {
		t.Fatalf("Long2Short out of order: mgpo-math=%d long2short=%d mgpo-code=%d", i, j, k)
	}
	// The Long2Short stage's provenance records the zero-sum reshape.
	assertProvenanceContains(t, out, "long2short", "zero-sum")
	// Instruct RL runs after distillation (DESIGN §4 terminal ordering).
	if d, ir := stageIndex(out, "distill"), stageIndex(out, "instruct-rl"); !(d < ir) {
		t.Fatalf("Instruct RL out of order: distill=%d instruct-rl=%d", d, ir)
	}
	assertProvenanceContains(t, out, "instruct-rl", "rule+rubric")
}

func stageIndex(c *ssp.Checkpoint, stage string) int {
	for i, p := range c.History {
		if p.Stage == stage {
			return i
		}
	}
	return -1
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
