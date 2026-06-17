package sft

import (
	"context"
	"fmt"
	"math"
)

// cosPi returns cos(π·t), used by the cosine learning-rate decay.
func cosPi(t float64) float64 { return math.Cos(math.Pi * t) }

// A Problem is one training item with the reference-rollout statistics the
// stage-2 hard-subset filter needs.
type Problem struct {
	ID string

	// TraceLen is the token length of the reference reasoning trace.
	TraceLen int

	// Rollouts is the number of reference rollouts attempted for this problem
	// (paper: 8 rollouts with VibeThinker-1.5B as the reference model).
	Rollouts int

	// Errors is how many of those rollouts were wrong. ErrorRate = Errors /
	// Rollouts; a high error rate marks a genuinely hard problem.
	Errors int
}

// ErrorRate returns the fraction of reference rollouts that were wrong, or 0
// when there were no rollouts.
func (p Problem) ErrorRate() float64 {
	if p.Rollouts <= 0 {
		return 0
	}
	return float64(p.Errors) / float64(p.Rollouts)
}

// A Filter selects the stage-2 hard-reasoning subset (DESIGN §4.1): keep only
// problems whose trace is at least MinTraceLen tokens and whose reference error
// rate is at least MinErrorRate. The paper's stage-2 values are MinTraceLen
// 5000 and MinErrorRate 0.75.
type Filter struct {
	MinTraceLen  int
	MinErrorRate float64
}

// DefaultHardFilter returns the paper's stage-2 hard-subset filter: drop traces
// shorter than 5000 tokens and problems with reference error rate below 0.75.
func DefaultHardFilter() Filter {
	return Filter{MinTraceLen: 5000, MinErrorRate: 0.75}
}

// Keep reports whether a problem passes the filter.
func (f Filter) Keep(p Problem) bool {
	return p.TraceLen >= f.MinTraceLen && p.ErrorRate() >= f.MinErrorRate
}

// Select returns the subset of problems the filter keeps, preserving order.
func (f Filter) Select(problems []Problem) []Problem {
	out := make([]Problem, 0, len(problems))
	for _, p := range problems {
		if f.Keep(p) {
			out = append(out, p)
		}
	}
	return out
}

// A Schedule is a CPU-side learning-rate schedule fed to the training loop
// per step. mlx-go-lm training takes a per-step lr as an input, so the cosine
// decay with linear warmup the papers specify is computed here and supplied.
type Schedule struct {
	Peak        float64 // peak learning rate after warmup
	Final       float64 // floor learning rate at the end of decay
	WarmupSteps int     // linear warmup steps from 0 to Peak
	TotalSteps  int     // total steps; cosine-decays Peak→Final after warmup
}

// LR returns the learning rate at zero-based step i: linear warmup to Peak over
// WarmupSteps, then cosine decay from Peak to Final across the remaining steps.
func (s Schedule) LR(i int) float64 {
	if i < 0 {
		i = 0
	}
	if s.WarmupSteps > 0 && i < s.WarmupSteps {
		return s.Peak * float64(i+1) / float64(s.WarmupSteps)
	}
	denom := s.TotalSteps - s.WarmupSteps
	if denom <= 0 {
		return s.Final
	}
	t := float64(i-s.WarmupSteps) / float64(denom)
	if t > 1 {
		t = 1
	}
	cos := 0.5 * (1 + cosPi(t))
	return s.Final + (s.Peak-s.Final)*cos
}

// A Stage describes one SFT stage of the curriculum.
type Stage struct {
	Name        string
	Epochs      int
	Schedule    Schedule
	HardFilter  *Filter // nil for broad coverage; set for the hard subset
	GlobalBatch int
	PackBlock   int // sequence-packing block size; 0 disables packing
}

// A Curriculum is the ordered list of SFT stages for a recipe. The 3B recipe is
// two stages (broad coverage, then hard subset).
type Curriculum struct {
	Stages []Stage
}

// DefaultCurriculum3B returns the paper's two-stage 3B SFT curriculum:
// stage 1 broad coverage (5 epochs), stage 2 hard subset (+2 epochs) with the
// default hard filter, both with global batch 128 and sequence packing.
func DefaultCurriculum3B(stage1Steps, stage2Steps int) Curriculum {
	hf := DefaultHardFilter()
	return Curriculum{Stages: []Stage{
		{
			Name:        "broad",
			Epochs:      5,
			GlobalBatch: 128,
			PackBlock:   4096,
			Schedule:    Schedule{Peak: 5e-5, Final: 8e-8, WarmupSteps: stage1Steps / 20, TotalSteps: stage1Steps},
		},
		{
			Name:        "hard",
			Epochs:      2,
			GlobalBatch: 128,
			PackBlock:   4096,
			HardFilter:  &hf,
			Schedule:    Schedule{Peak: 5e-5, Final: 8e-8, WarmupSteps: stage2Steps / 20, TotalSteps: stage2Steps},
		},
	}}
}

// A Trainer runs one SFT stage over its problems and reports where the stage's
// checkpoint was written. It is the seam by which the heavy mlx-go-lm full
// fine-tune plugs in; tests inject a fake.
type Trainer interface {
	// TrainStage trains inDir's weights on problems under the stage's
	// hyperparameters and returns the output checkpoint directory.
	TrainStage(ctx context.Context, stage Stage, inDir string, problems []Problem) (outDir string, err error)
}

// Run executes the curriculum stages in order, threading the checkpoint
// directory and applying each stage's hard filter to the problem set before
// training. It returns the final checkpoint directory.
func (c Curriculum) Run(ctx context.Context, tr Trainer, inDir string, problems []Problem) (string, error) {
	if tr == nil {
		return "", fmt.Errorf("sft: nil trainer")
	}
	cur := inDir
	for i, st := range c.Stages {
		if err := ctx.Err(); err != nil {
			return "", fmt.Errorf("sft: curriculum canceled before stage %q: %w", st.Name, err)
		}
		data := problems
		if st.HardFilter != nil {
			data = st.HardFilter.Select(problems)
		}
		out, err := tr.TrainStage(ctx, st, cur, data)
		if err != nil {
			return "", fmt.Errorf("sft: stage %d %q: %w", i, st.Name, err)
		}
		cur = out
	}
	return cur, nil
}
