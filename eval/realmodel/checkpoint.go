//go:build modelir

package realmodel

import (
	"fmt"

	"github.com/tmc/mlx-go/mlx"
)

// trainedQVMeta marks a checkpoint as the bounded trained set (q/v projections
// only), so loadTrainedQV can refuse a mismatched file rather than silently
// applying the wrong tensors.
const trainedQVMeta = "vibethinker-realmodel-trained-qv-v1"

// saveTrainedQV writes the model's current trained-set weights (the q/v
// projections — the bounded set the optimizer touches, see trainedKey) to a
// safetensors file, keyed by their parameter path. It is the bridge across the
// two-process split: the train child saves it after training, the score child
// applies it before the final held-out pass. Only ~tens of MB (two matrices per
// layer), not the multi-GB model.
//
// The arrays are written through the SAME slot Get the trainer reads, and reload
// applies them through the SAME slot Set the trainer writes, so the round-trip is
// symmetric — no HF-key or transpose translation is needed (that is only required
// when crossing into the on-disk HF layout, which this private checkpoint does
// not use).
func saveTrainedQV(m *Model, path string) (int, error) {
	slots, vals, err := snapshotParams(m)
	if err != nil {
		return 0, fmt.Errorf("realmodel: checkpoint snapshot: %w", err)
	}
	defer func() {
		for _, v := range vals {
			v.Free()
		}
	}()
	tensors := make(map[string]*mlx.Array, len(slots))
	for i, s := range slots {
		tensors[s.path] = vals[i]
	}
	if err := mlx.SaveSafetensors(path, tensors, map[string]string{"kind": trainedQVMeta}); err != nil {
		return 0, fmt.Errorf("realmodel: save checkpoint %q: %w", path, err)
	}
	return len(tensors), nil
}

// loadTrainedQV applies a saved trained-set checkpoint to the model, overwriting
// its q/v projections with the trained values, through the slot Set hooks (the
// same bias-collision-free write path the trainer uses). It verifies the file is
// a trained-qv checkpoint and that every model q/v slot has a matching tensor, so
// a wrong or partial file fails loudly rather than half-applying.
func loadTrainedQV(m *Model, path string) (int, error) {
	tensors, meta, err := mlx.LoadSafetensors(path)
	if err != nil {
		return 0, fmt.Errorf("realmodel: load checkpoint %q: %w", path, err)
	}
	defer func() {
		for _, a := range tensors {
			a.Free()
		}
	}()
	if meta["kind"] != trainedQVMeta {
		return 0, fmt.Errorf("realmodel: checkpoint %q is not a %s (kind=%q)", path, trainedQVMeta, meta["kind"])
	}
	slots, _, err := collectSlots(m)
	if err != nil {
		return 0, fmt.Errorf("realmodel: checkpoint collect slots: %w", err)
	}
	for _, s := range slots {
		a, ok := tensors[s.path]
		if !ok {
			return 0, fmt.Errorf("realmodel: checkpoint %q missing tensor for slot %q", path, s.path)
		}
		// Apply a copy so the model owns its weight independently of the loaded map
		// (which is freed on return).
		if err := s.set(mlx.Copy(a)); err != nil {
			return 0, fmt.Errorf("realmodel: checkpoint set %q: %w", s.path, err)
		}
	}
	return len(slots), nil
}
