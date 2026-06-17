package fuse

import (
	"fmt"

	"github.com/tmc/mlx-go/mlx"
)

// MergeFiles loads each safetensors file in paths, fuses them with the given
// weights, and writes the merged tensors to outPath. Weights follow the same
// rules as [Merge]; pass [UniformWeights](len(paths)) for the paper default.
//
// The merged file carries metadata recording that it is a fusion and how many
// experts contributed.
func MergeFiles(paths []string, weights []float64, outPath string) error {
	if len(paths) == 0 {
		return fmt.Errorf("fuse: no input paths")
	}
	models := make([]Model, len(paths))
	for i, p := range paths {
		tensors, _, err := mlx.LoadSafetensors(p)
		if err != nil {
			return fmt.Errorf("fuse: loading %q: %w", p, err)
		}
		models[i] = Model{Name: p, Tensors: tensors}
	}
	merged, err := Merge(models, weights)
	if err != nil {
		return err
	}
	meta := map[string]string{
		"fused":         "expert-model-fusion",
		"fused_experts": fmt.Sprintf("%d", len(paths)),
	}
	if err := mlx.SaveSafetensors(outPath, merged, meta); err != nil {
		return fmt.Errorf("fuse: saving merged model to %q: %w", outPath, err)
	}
	return nil
}
