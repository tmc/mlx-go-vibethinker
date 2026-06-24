//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/training"
	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
)

// Method is the method-comparison configuration reused verbatim from
// eval/methodcompare, so the real-model run exercises exactly the same nine
// named configurations (baseline, each Tier-1/2/3 refinement, all-on) as the
// toy harness.
type Method = methodcompare.Method

// paramSlot is one trainable linear-weight parameter, addressed through its
// visitor binding's Set hook rather than a SetWeights path map. The model's
// SetWeights keys weight and bias bindings by the SAME path (Qwen2 attention
// projections carry biases), so the bias binding — which has no Set — clobbers
// the weight binding in the path map and every weight reads back "not writable".
// Writing through the binding's Set directly sidesteps that path collision.
type paramSlot struct {
	path string
	set  func(*mlx.Array) error
}

// trainer wraps a real model for full-parameter GRPO optimizer steps. It holds
// the trainable param slots in a stable (sorted) order so the param vector and
// the slots stay aligned across the value-and-grad closure and the optimizer
// step.
type trainer struct {
	m     *Model
	slots []paramSlot
	opt   *training.SeparateVGAndOptimizer
	lr    *mlx.Array
}

// trainedKey reports whether a parameter path is in the bounded set the smoke
// optimizes. We train only the attention query/value projections — the canonical
// minimal trainable set (the substrate's default TrainableKeys) — NOT all 196
// weight matrices of the 1.5B. Full-rank training of every matrix would allocate
// ~10GB of fp32 Adam moments per method, and the 9-method x 2-source loop would
// exhaust memory. The bounded set is still a genuine full-precision optimizer
// step that makes current diverge from old (the ratio evidence holds); the smoke
// validates the mechanism, not convergence.
func trainedKey(path string) bool {
	return strings.HasSuffix(path, ".self_attn.q_proj") || strings.HasSuffix(path, ".self_attn.v_proj")
}

// collectSlots visits the model and returns the writable linear-weight params in
// the bounded trained set (q/v projections), with their values and Set hooks, in
// sorted-by-path order. It skips biases, norms, the tied embedding, and the
// untrained projections/MLP matrices.
func collectSlots(m *Model) ([]paramSlot, []*mlx.Array, error) {
	pv, ok := m.LM.(models.ParamVisitable)
	if !ok {
		return nil, nil, fmt.Errorf("realmodel: model is not ParamVisitable")
	}
	type entry struct {
		slot paramSlot
		val  *mlx.Array
	}
	var entries []entry
	seen := map[string]bool{}
	pv.VisitParams(func(ref models.ParamRef, b models.ParamBinding) {
		if ref.Kind != models.ParamLinearWeight || b.Set == nil || b.Get == nil {
			return
		}
		if seen[ref.Path] || !trainedKey(ref.Path) {
			return
		}
		val := b.Get()
		if val == nil {
			return
		}
		seen[ref.Path] = true
		entries = append(entries, entry{slot: paramSlot{path: ref.Path, set: b.Set}, val: val})
	})
	if len(entries) == 0 {
		return nil, nil, fmt.Errorf("realmodel: no writable trained-set params")
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].slot.path < entries[j].slot.path })
	slots := make([]paramSlot, len(entries))
	vals := make([]*mlx.Array, len(entries))
	for i, e := range entries {
		slots[i] = e.slot
		vals[i] = e.val
	}
	return slots, vals, nil
}

// writeParams writes the param vector back into the model through the slot Set
// hooks (bypassing the path-keyed SetWeights collision).
func (t *trainer) writeParams(params []*mlx.Array) error {
	for i, s := range t.slots {
		if err := s.set(params[i]); err != nil {
			return fmt.Errorf("realmodel: set %s: %w", s.path, err)
		}
	}
	return nil
}

// snapshotParams returns evaluated deep copies of the model's current writable
// linear-weight params, in slot order — a base-policy snapshot that restore can
// write back so successive methods start from the same weights WITHOUT reloading
// the multi-GB model (reloading a second model OOMs the wired working set).
func snapshotParams(m *Model) ([]paramSlot, []*mlx.Array, error) {
	slots, vals, err := collectSlots(m)
	if err != nil {
		return nil, nil, err
	}
	snap := make([]*mlx.Array, len(vals))
	for i, v := range vals {
		snap[i] = mlx.Copy(v)
	}
	if err := mlx.Eval(snap...); err != nil {
		return nil, nil, fmt.Errorf("realmodel: eval param snapshot: %w", err)
	}
	return slots, snap, nil
}

// restoreParams writes a base-policy snapshot back into the model through the
// slot Set hooks, returning the model to the snapshot's weights.
func restoreParams(slots []paramSlot, snap []*mlx.Array) error {
	for i, s := range slots {
		if err := s.set(mlx.Copy(snap[i])); err != nil {
			return fmt.Errorf("realmodel: restore %s: %w", s.path, err)
		}
	}
	return nil
}

// newTrainer collects the model's writable linear-weight params and builds a
// real (separate value-and-grad + Adam) optimizer over them. The lossFn writes
// the live params into the model before the forward inside lossClosure, so the
// gradient flows back to them (the same SetWeights-before-forward bridge
// training/full.go uses, but through the binding Set hooks).
func newTrainer(m *Model, lr float64, lossClosure func() (*mlx.Array, error)) (*trainer, error) {
	slots, params, err := collectSlots(m)
	if err != nil {
		return nil, err
	}
	t := &trainer{m: m, slots: slots, lr: mlx.NewScalar(float32(lr))}

	lossFn := func(ctx context.Context, modelParams, _ []*mlx.Array) (*mlx.Array, error) {
		if err := t.writeParams(modelParams); err != nil {
			return nil, err
		}
		return lossClosure()
	}

	t.opt = training.NewSeparateVGAndOptimizer(len(params), float32(lr), lossFn)
	t.opt.InitState(params)
	return t, nil
}

// step runs one real value-and-grad + Adam update and returns the scalar loss
// value for this step. The model's weights are updated in place.
func (t *trainer) step() (float64, error) {
	loss := t.opt.Step(nil, nil, t.lr)
	if loss == nil {
		return 0, fmt.Errorf("realmodel: nil loss from optimizer step")
	}
	v := mlx.ArrayItemFloat32(loss)
	loss.Free()
	// Commit the updated params back into the model so a subsequent rollout or
	// rescore (outside the grad closure) sees the new policy.
	if err := t.writeParams(t.opt.GetParams()); err != nil {
		return 0, err
	}
	return float64(v), nil
}

// commitDurable writes model-owned deep copies of the optimizer's current params
// back into the model, so the model no longer aliases arrays the optimizer owns.
// It MUST be called before free() whenever the model is forwarded again after
// training: SeparateVGAndOptimizer.Free frees s.Params, and those Params are the
// exact arrays the trainer wrote into the model's q/v slots — so without this
// detach the next Forward dereferences freed weights (panic: b is nil). The
// per-method Evaluate paths reload a fresh model and never forward post-train, so
// they do not need it; the sweep, which scores held-out on the trained policy,
// does.
func (t *trainer) commitDurable() error {
	params := t.opt.GetParams()
	if len(params) != len(t.slots) {
		return fmt.Errorf("realmodel: commitDurable param/slot mismatch %d != %d", len(params), len(t.slots))
	}
	copies := make([]*mlx.Array, len(params))
	for i, p := range params {
		copies[i] = mlx.Copy(p)
	}
	if err := mlx.Eval(copies...); err != nil {
		return fmt.Errorf("realmodel: commitDurable eval: %w", err)
	}
	for i, s := range t.slots {
		if err := s.set(copies[i]); err != nil {
			return fmt.Errorf("realmodel: commitDurable set %s: %w", s.path, err)
		}
	}
	return nil
}

// free releases the optimizer resources.
func (t *trainer) free() {
	if t.opt != nil {
		t.opt.Free()
	}
	if t.lr != nil {
		t.lr.Free()
	}
}

// baselineClipEps is the symmetric baseline clip epsilon. As in the toy
// harness, the DESIGN.md baseline clips symmetrically — rl.DefaultGRPOConfig
// ships the asymmetric Clip-Higher values (0.2/0.28), which would make the
// +ClipHigher row indistinguishable from baseline. We build from a symmetric
// base and let the +ClipHigher method's Options override it.
const baselineClipEps = 0.2

// baseConfig returns the symmetric baseline GRPOConfig the real run layers each
// method's Options onto.
func baseConfig() rl.GRPOConfig {
	cfg := rl.DefaultGRPOConfig()
	cfg.ClipEps = baselineClipEps
	cfg.ClipEpsLow = baselineClipEps
	cfg.ClipEpsHigh = baselineClipEps
	return cfg
}
