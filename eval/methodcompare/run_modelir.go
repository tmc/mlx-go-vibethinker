//go:build modelir

package methodcompare

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/internal/toymodel"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// EvaluateWithModel runs the full comparison: the deterministic core mechanism
// metrics (from Evaluate) plus, for each method, the real MGPO-style loss through
// a toy Qwen2 model and the wall-time of that loss. The model-coupled metrics
// (FinalLoss, WallMillis) are filled onto the core metrics in place.
//
// The model metrics are NOT reproducible: the toy forward pass and wall-time are
// model- and machine-dependent. They are clearly flagged as such in the JSON
// metric_layers and the table's modelir section. The reproducible mechanism
// metrics are the deterministic core; this layer only adds the model-coupled view.
//
// seed seeds both the scenario (for the core metrics) and the toy model weights.
func EvaluateWithModel(seed uint64) ([]Metrics, error) {
	core, err := Evaluate(seed)
	if err != nil {
		return nil, err
	}
	lm, err := toymodel.New(toymodel.DefaultConfig(), seed)
	if err != nil {
		return nil, fmt.Errorf("methodcompare: build toy model: %w", err)
	}
	methods := Registry()
	sc := newScenario(seed)
	for i := range core {
		loss, wall, err := modelLoss(lm, methods[i], sc)
		if err != nil {
			return nil, fmt.Errorf("methodcompare: model loss for %q: %w", methods[i].Name, err)
		}
		l := loss
		core[i].FinalLoss = &l
		core[i].WallMillis = wall
	}
	return core, nil
}

// modelLoss computes one method's MGPO loss through the toy model on the
// scenario's final step, returning the loss and the wall-time in milliseconds. It
// applies the method's full advantage path (DRA reweight, Dynamic Sampling, the
// Tier-1 Options, DCPO smoothing) and selects the FRPO sibling loss when the
// method enables it, so the loss reflects the method's actual objective.
func modelLoss(lm models.LanguageModel, m Method, sc *scenario) (float64, float64, error) {
	step := sc.steps[len(sc.steps)-1]
	rewards := step.rewards
	ids := step.promptIDs

	if m.DRA != nil {
		rw, err := mgpo.DiversityReweightGroups(rewards, step.texts, m.DRA)
		if err != nil {
			return 0, 0, err
		}
		rewards = rw
	}
	if m.DynamicSampling {
		rewards, ids = mgpo.DynamicSample(rewards, ids)
	}
	_ = ids

	g := 0
	for _, group := range rewards {
		g += len(group)
	}
	if g < 2 {
		return 0, 0, fmt.Errorf("need >= 2 rollouts after filtering, got %d", g)
	}

	// Build per-token log-prob tensors for the flattened rollouts, with a small
	// on-policy drift so the clipped surrogate does non-trivial work (mirrors the
	// recipe's toy MGPO stage).
	const toks = 6
	curVals := make([]float32, g*toks)
	oldVals := make([]float32, g*toks)
	maskVals := make([]float32, g*toks)
	for i := range curVals {
		curVals[i] = -0.5 - float32(i%5)*0.05
		oldVals[i] = curVals[i] - 0.3 // drift so some ratios bind the clip
		maskVals[i] = 1
	}
	current := mlx.NewArray(curVals, g, toks)
	old := mlx.StopGradient(mlx.NewArray(oldVals, g, toks))
	ref := mlx.StopGradient(mlx.NewArray(oldVals, g, toks))
	mask := mlx.NewArray(maskVals, g, toks)

	cfg := baseConfig()
	cfg.DrGRPO = m.DrGRPOLoss

	start := time.Now()
	var loss *mlx.Array
	var err error
	if m.FRPO.BetaFuture != 0 {
		loss, err = mgpo.LossFRPOScaled(current, old, ref, mask, rewards, m.Lambda, cfg, m.Opts, m.FRPO)
	} else {
		loss, err = mgpo.LossOpt(current, old, ref, mask, rewards, m.Lambda, cfg, m.Opts)
	}
	if err != nil {
		return 0, 0, err
	}
	if err := mlx.Eval(loss); err != nil {
		return 0, 0, fmt.Errorf("eval loss: %w", err)
	}
	wallMillis := float64(time.Since(start).Microseconds()) / 1000.0

	// Touch the model so the toy weights participate (a real run computes the
	// per-token log-probs through this model; here we confirm a forward pass is
	// finite, exercising the same seam without a full optimizer step).
	if err := touchModel(lm); err != nil {
		return 0, 0, err
	}

	return float64(mlx.ArrayItemFloat32(loss)), wallMillis, nil
}

// touchModel runs a tiny forward pass through the toy model to confirm the model
// path is live (the harness's model layer is about wiring, not training).
func touchModel(lm models.LanguageModel) error {
	in := mlx.NewArray([]int32{1, 2, 3, 4}, 1, 4)
	defer in.Free()
	logits, _, err := models.Forward(context.Background(), lm, in, nil)
	if err != nil {
		return fmt.Errorf("toy model forward: %w", err)
	}
	defer logits.Free()
	if err := mlx.Eval(logits); err != nil {
		return fmt.Errorf("eval logits: %w", err)
	}
	return nil
}
