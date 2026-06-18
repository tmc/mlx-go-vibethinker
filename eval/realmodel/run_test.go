//go:build modelir

package realmodel

import (
	"context"
	"testing"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
)

// smokeConfig is a minimal real-GRPO config: few prompts, K>=4 rollouts, short
// completions, a few real optimizer steps — enough to drive the full
// rollout->rescore->real-step pipeline to completion on the real 1.5B without
// being a convergence run.
func smokeConfig() Config {
	return Config{
		Prompts:     3,
		K:           4,
		MaxTokens:   24,
		Temperature: 0.9,
		Steps:       3,
		LR:          1e-6,
		Seed:        1,
	}
}

// The baseline method must run the full real GRPO loop to completion on the real
// model with a finite, non-diverging loss, AND the importance ratio must be a
// real distribution (var > 0, not a delta at 1) after the optimizer steps —
// the direct evidence the generate->rescore->step wiring is live and old was NOT
// rescored from post-update weights.
func TestBaselineRealGRPOLoopRunsAndRatioIsReal(t *testing.T) {
	m := requireModel(t)
	method := methodcompare.Registry()[0] // baseline
	if method.Name != "baseline" {
		t.Fatalf("registry[0] = %q, want baseline", method.Name)
	}

	ctx := context.Background()
	groups, err := buildOrganicGroups(ctx, m, smokeConfig())
	if err != nil {
		t.Fatalf("buildOrganicGroups: %v", err)
	}
	mt, err := runMethod(ctx, m, method, smokeConfig(), groups)
	if err != nil {
		t.Fatalf("runMethod(baseline): %v", err)
	}

	if mt.Steps == 0 {
		t.Fatal("baseline took no optimizer steps")
	}
	if mt.FinalLoss == nil {
		t.Fatal("baseline FinalLoss is nil after a real run")
	}
	if !mt.LossFinite {
		t.Fatal("baseline loss not finite over the run")
	}
	// The rescore must be real: a non-trivial ratio distribution after >=1 step.
	if mt.RatioVar <= 0 {
		t.Fatalf("importance ratio var = %v, want > 0 (rescore collapsed to a delta at 1 — old likely rescored from post-update weights)", mt.RatioVar)
	}
	if mt.RatioMaxAbsDev <= 0 {
		t.Fatalf("max|ratio-1| = %v, want > 0", mt.RatioMaxAbsDev)
	}
	t.Logf("baseline real GRPO: steps=%d finalLoss=%.5g maxAbsLoss=%.5g ratioMean=%.5f ratioVar=%.3g maxAbsDev=%.3g acc=%.2f cliff=%d learn=%d/%d",
		mt.Steps, *mt.FinalLoss, mt.MaxAbsLoss, mt.RatioMean, mt.RatioVar, mt.RatioMaxAbsDev,
		mt.AccMean, mt.CliffGroups, mt.LearnGroups, mt.GroupsTotal)
}
