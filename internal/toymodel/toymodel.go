package toymodel

import (
	"fmt"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"
)

// Config describes a toy Qwen2. The defaults from DefaultConfig are tiny enough
// to evaluate on CPU.
type Config struct {
	Vocab        int
	Hidden       int
	Layers       int
	Heads        int
	Intermediate int
}

// DefaultConfig returns a 2-layer toy Qwen2 configuration.
func DefaultConfig() Config {
	return Config{Vocab: 256, Hidden: 32, Layers: 2, Heads: 2, Intermediate: 64}
}

// New builds a toy Qwen2 model with deterministic weights for the given seed.
func New(cfg Config, seed uint64) (models.LanguageModel, error) {
	sc := models.SyntheticConfig("qwen2", cfg.Vocab, cfg.Hidden, cfg.Layers, cfg.Heads, cfg.Intermediate,
		func(m map[string]any) {
			m["num_key_value_heads"] = cfg.Heads
			m["rms_norm_eps"] = 1e-6
			m["rope_theta"] = 10000.0
			m["tie_word_embeddings"] = true
		})
	lm, err := models.NewSyntheticModel("qwen2", sc, seed)
	if err != nil {
		return nil, fmt.Errorf("toymodel: build synthetic qwen2: %w", err)
	}
	return lm, nil
}

// Weights extracts the model's parameters as a HuggingFace-keyed name→array map
// reloadable by [LoadInto] via the model's LoadWeights. Two conversions bridge
// the model's in-memory representation to the on-disk HuggingFace format:
//
//   - keys gain a kind-dependent suffix (".bias" for biases, ".weight"
//     otherwise), because the param visitor reports canonical paths but
//     LoadWeights reads HF keys;
//   - linear weights are transposed from the model's native [in, out] storage
//     back to HuggingFace [out, in], because LoadWeights transposes them on the
//     way in (StandardLinear stores pre-transposed weights for mlx.Matmul).
//
// With both conversions, Save → LoadInto reproduces the source model exactly.
func Weights(lm models.LanguageModel) (map[string]*mlx.Array, error) {
	pv, ok := lm.(models.ParamVisitable)
	if !ok {
		return nil, fmt.Errorf("toymodel: model is not ParamVisitable")
	}
	out := map[string]*mlx.Array{}
	pv.VisitParams(func(ref models.ParamRef, b models.ParamBinding) {
		if b.Get == nil {
			return
		}
		a := b.Get()
		if a == nil {
			return
		}
		if ref.Kind == models.ParamLinearWeight && len(a.Shape()) == 2 {
			a = mlx.Transpose(a)
		}
		out[hfKey(ref)] = a
	})
	if len(out) == 0 {
		return nil, fmt.Errorf("toymodel: model produced no weights")
	}
	return out, nil
}

// hfKey maps a parameter's canonical path to its on-disk HuggingFace key by
// appending the suffix LoadWeights expects for the parameter's kind.
func hfKey(ref models.ParamRef) string {
	if ref.Kind == models.ParamLinearBias {
		return ref.Path + ".bias"
	}
	return ref.Path + ".weight"
}

// Save writes the model's weights to a safetensors file at path.
func Save(lm models.LanguageModel, path string) error {
	w, err := Weights(lm)
	if err != nil {
		return err
	}
	if err := mlx.Eval(values(w)...); err != nil {
		return fmt.Errorf("toymodel: eval weights: %w", err)
	}
	if err := mlx.SaveSafetensors(path, w, map[string]string{"toy": "qwen2"}); err != nil {
		return fmt.Errorf("toymodel: save %q: %w", path, err)
	}
	return nil
}

// weightLoader is the model's canonical weight-loading hook
// (Qwen2Model.LoadWeights). It reads HuggingFace-format safetensors, transposes
// linear weights into the model's native layout, and restores every layer —
// including the tied embedding, whose param-binding setter is a no-op.
type weightLoader interface {
	LoadWeights(weightFiles ...string) error
}

// LoadInto overwrites a model's parameters from a safetensors file written by
// [Save], using the model's own LoadWeights path. The file must come from a
// structurally identical model.
func LoadInto(lm models.LanguageModel, path string) error {
	wl, ok := lm.(weightLoader)
	if !ok {
		return fmt.Errorf("toymodel: model does not support LoadWeights")
	}
	if err := wl.LoadWeights(path); err != nil {
		return fmt.Errorf("toymodel: load %q: %w", path, err)
	}
	return nil
}

func values(m map[string]*mlx.Array) []*mlx.Array {
	out := make([]*mlx.Array, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}
