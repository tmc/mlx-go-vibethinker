package fuse

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/mlx-go/mlx"
)

// weightTolerance is the slack allowed when checking that fusion weights sum to
// 1. Weights are user-supplied floats, so exact equality is too strict.
const weightTolerance = 1e-6

// A Model is one expert to be fused: a name (for diagnostics) and its parameter
// tensors keyed by tensor name. Tensors is typically the result of
// mlx.LoadSafetensors.
type Model struct {
	Name    string
	Tensors map[string]*mlx.Array
}

// UniformWeights returns the paper-default fusion weights: n equal weights of
// 1/n. It returns nil for n <= 0.
func UniformWeights(n int) []float64 {
	if n <= 0 {
		return nil
	}
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}
	return w
}

// Merge fuses the given expert models into one by a weighted per-tensor average
// of their parameters, returning the merged tensor map. The result casts each
// tensor back to its source dtype after averaging in float32.
//
// Merge validates, then delegates to the unexported core:
//   - models must be non-empty and weights must have one entry per model;
//   - weights must be non-negative and sum to 1 within weightTolerance;
//   - all models must agree on tensor names, shapes, and dtypes.
//
// Any disagreement is an error; Merge never drops or zero-fills a tensor.
func Merge(models []Model, weights []float64) (map[string]*mlx.Array, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("fuse: no models to merge")
	}
	if len(weights) != len(models) {
		return nil, fmt.Errorf("fuse: have %d models but %d weights", len(models), len(weights))
	}
	if err := validateWeights(weights); err != nil {
		return nil, err
	}
	if err := validateAgreement(models); err != nil {
		return nil, err
	}
	return merge(models, weights)
}

// validateWeights checks non-negativity and normalization.
func validateWeights(weights []float64) error {
	var sum float64
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("fuse: weight %d is not finite: %v", i, w)
		}
		if w < 0 {
			return fmt.Errorf("fuse: weight %d is negative: %v", i, w)
		}
		sum += w
	}
	if math.Abs(sum-1.0) > weightTolerance {
		return fmt.Errorf("fuse: weights sum to %v, want 1 (within %g)", sum, weightTolerance)
	}
	return nil
}

// validateAgreement checks that every model has the same tensor names, and that
// each tensor's shape and dtype agree across models. The first model defines
// the reference set.
func validateAgreement(models []Model) error {
	ref := models[0]
	if len(ref.Tensors) == 0 {
		return fmt.Errorf("fuse: model %q has no tensors", modelName(ref, 0))
	}
	refShapes := make(map[string][]int, len(ref.Tensors))
	refDtypes := make(map[string]mlx.Dtype, len(ref.Tensors))
	for name, t := range ref.Tensors {
		if t == nil {
			return fmt.Errorf("fuse: model %q tensor %q is nil", modelName(ref, 0), name)
		}
		refShapes[name] = t.Shape()
		refDtypes[name] = t.Dtype()
	}
	for i, m := range models {
		if len(m.Tensors) != len(ref.Tensors) {
			return fmt.Errorf("fuse: model %q has %d tensors, model %q has %d",
				modelName(m, i), len(m.Tensors), modelName(ref, 0), len(ref.Tensors))
		}
		for name, t := range m.Tensors {
			if t == nil {
				return fmt.Errorf("fuse: model %q tensor %q is nil", modelName(m, i), name)
			}
			refShape, ok := refShapes[name]
			if !ok {
				return fmt.Errorf("fuse: model %q has tensor %q not present in model %q",
					modelName(m, i), name, modelName(ref, 0))
			}
			if !shapeEqual(t.Shape(), refShape) {
				return fmt.Errorf("fuse: tensor %q shape mismatch: model %q has %v, model %q has %v",
					name, modelName(m, i), t.Shape(), modelName(ref, 0), refShape)
			}
			if t.Dtype() != refDtypes[name] {
				return fmt.Errorf("fuse: tensor %q dtype mismatch: model %q has %v, model %q has %v",
					name, modelName(m, i), t.Dtype(), modelName(ref, 0), refDtypes[name])
			}
		}
	}
	return nil
}

// merge is the averaging core. It assumes inputs have already been validated:
// non-empty models, matching weight count, normalized non-negative weights, and
// full tensor-name/shape/dtype agreement. For each tensor it accumulates
// Σ wᵢ·Mᵢ in float32 and casts the result back to the source dtype.
func merge(models []Model, weights []float64) (map[string]*mlx.Array, error) {
	out := make(map[string]*mlx.Array, len(models[0].Tensors))
	// Sort names for deterministic evaluation order.
	names := make([]string, 0, len(models[0].Tensors))
	for name := range models[0].Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		dtype := models[0].Tensors[name].Dtype()
		var acc *mlx.Array
		for i, m := range models {
			t := mlx.Astype(m.Tensors[name], mlx.Float32)
			term := mlx.MultiplyScalar(t, float32(weights[i]))
			if acc == nil {
				acc = term
			} else {
				acc = mlx.Add(acc, term)
			}
		}
		// Cast back to the source dtype to preserve storage format.
		out[name] = mlx.Astype(acc, dtype)
	}
	if err := mlx.Eval(collect(out)...); err != nil {
		return nil, fmt.Errorf("fuse: evaluating merged tensors: %w", err)
	}
	return out, nil
}

func collect(m map[string]*mlx.Array) []*mlx.Array {
	out := make([]*mlx.Array, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}

func modelName(m Model, i int) string {
	if m.Name != "" {
		return m.Name
	}
	return fmt.Sprintf("#%d", i)
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
