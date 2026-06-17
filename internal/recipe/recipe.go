//go:build modelir

package recipe

import "github.com/tmc/mlx-go-vibethinker/ssp"

// toyData is a handful of toy training strings.
var toyData = []string{
	"the cat sat on the mat",
	"two plus two equals four",
	"the quick brown fox jumps over the lazy dog",
	"a stitch in time saves nine",
}

// toyTraces is a handful of toy reasoning traces for distillation.
var toyTraces = []string{
	"first we note that x is two then y is three so the answer is five",
	"by symmetry the triangle is isosceles hence the base angles are equal",
	"compute the derivative set it to zero and solve for the critical point",
}

// Pipeline15B builds the toy 1.5B recipe: SFT spectrum (probe + fuse) then MGPO
// math then MGPO code.
func (t *Toy) Pipeline15B() *ssp.Pipeline {
	return &ssp.Pipeline{
		Name: "vibethinker-1.5b",
		Stages: []ssp.Stage{
			t.SFTStage("sft", toyData),
			t.ProbeStage("probe"),
			t.FuseStage("fuse", 4),
			t.MGPOStage("mgpo-math", "solve x", []float64{1, 0, 1, 1}, 1.0),
			t.MGPOStage("mgpo-code", "write fn", []float64{1, 1, 0, 0}, 1.0),
		},
	}
}

// Pipeline3B builds the toy 3B recipe: two-stage curriculum SFT (each with probe
// + fuse), then MGPO math, Long2Short math, MGPO code, MGPO STEM, offline
// self-distillation, and Instruct RL — the full DESIGN §4 ordering. The real 3B
// uses a single 64K context window (the paper found progressive truncation hurt
// the stronger 3B init); that window size is a gated real-run detail, not
// exercised on the CPU toy path.
func (t *Toy) Pipeline3B() *ssp.Pipeline {
	return &ssp.Pipeline{
		Name: "vibethinker-3b",
		Stages: []ssp.Stage{
			t.SFTStage("sft-broad", toyData),
			t.ProbeStage("probe-broad"),
			t.FuseStage("fuse-broad", 4),
			t.SFTStage("sft-hard", toyData),
			t.ProbeStage("probe-hard"),
			t.FuseStage("fuse-hard", 4),
			t.MGPOStage("mgpo-math", "solve x", []float64{1, 0, 1, 1}, 1.0),
			t.Long2ShortStage("long2short"),
			t.MGPOStage("mgpo-code", "write fn", []float64{1, 1, 1, 0}, 1.0),
			t.MGPOStage("mgpo-stem", "explain", []float64{1, 0, 0, 0}, 1.0),
			t.DistillStage("distill", toyTraces),
			t.InstructStage("instruct-rl"),
		},
	}
}
