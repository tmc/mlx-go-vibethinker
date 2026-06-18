//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/training"
	"github.com/tmc/mlx-go-lm/mlxlm/loader"
	"github.com/tmc/mlx-go/mlx"
)

// modelDirEnv overrides the model directory the smoke test loads from.
const modelDirEnv = "VIBETHINKER_REALMODEL_DIR"

// DefaultModelDir returns the directory the real-model smoke test loads from:
// the VIBETHINKER_REALMODEL_DIR environment variable if set, otherwise the
// conventional download location under the user's home directory. The directory
// must be a HuggingFace model export (config.json, *.safetensors, tokenizer.json)
// for Qwen2.5-Math-1.5B (DESIGN.md's 1.5B target). The weights are multi-GB and
// are never committed.
func DefaultModelDir() string {
	if d := os.Getenv(modelDirEnv); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "models-tmp", "Qwen2.5-Math-1.5B")
}

// Model is a loaded real language model and its tokenizer, the substrate for the
// real-model smoke test. The model satisfies both models.LanguageModel (for the
// forward pass) and training.Trainable (for the real optimizer step).
type Model struct {
	LM    models.LanguageModel
	Tok   mlxlm.Tokenizer
	Dir   string
	Tr    training.Trainable
	vocab int
}

// Load loads the real model and tokenizer from a HuggingFace directory. It
// returns an error (not a panic) when the directory is missing or the model is
// not trainable, so callers — including tests that skip when the model is not
// present — can handle it gracefully.
func Load(ctx context.Context, dir string) (*Model, error) {
	if dir == "" {
		return nil, fmt.Errorf("realmodel: empty model directory (set %s)", modelDirEnv)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return nil, fmt.Errorf("realmodel: no config.json in %q (download Qwen2.5-Math-1.5B or set %s): %w", dir, modelDirEnv, err)
	}

	// loader.LoadModel resolves the path, constructs the architecture, discovers
	// the safetensors, AND calls LoadWeights to populate them — models.LoadModel
	// alone only builds the (empty-weight) architecture.
	bundle, err := loader.LoadModel(ctx, dir, loader.LocalResolver(nil), loader.LoadOptions{})
	if err != nil {
		return nil, fmt.Errorf("realmodel: load model from %q: %w", dir, err)
	}
	lm := bundle.Model
	tr, ok := lm.(training.Trainable)
	if !ok {
		return nil, fmt.Errorf("realmodel: model from %q is not training.Trainable (no Weights/SetWeights)", dir)
	}

	tok, err := mlxlm.LoadTokenizer(dir)
	if err != nil {
		return nil, fmt.Errorf("realmodel: load tokenizer from %q: %w", dir, err)
	}

	// We deliberately do NOT raise the wired limit to the device max: wiring the
	// whole 1.5B model pins memory and starves the full-parameter backward
	// (gradients + Adam moments), which OOMs. Let MLX manage residency; the
	// per-process array ceiling on long rollouts is handled by closing KV caches
	// and clearing the buffer cache between rollouts (see rollout.go).

	return &Model{LM: lm, Tok: tok, Dir: dir, Tr: tr, vocab: tok.VocabSize()}, nil
}

// Vocab is the tokenizer's vocabulary size — the expected last dimension of the
// model's logits.
func (m *Model) Vocab() int { return m.vocab }

// Forward runs the model on a single token sequence and returns the
// materialized logits over the whole sequence, shape [1, len(ids), vocab]. It
// uses models.ForwardAndSync so the returned array is safe to read and free
// independently (the smoke and rescore paths both want materialized logits).
func (m *Model) Forward(ctx context.Context, ids []int32) (*mlx.Array, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("realmodel: empty token sequence")
	}
	in := mlx.NewArray(ids, 1, len(ids))
	defer in.Free()
	logits, cache, err := models.ForwardAndSync(ctx, m.LM, in, nil)
	if err != nil {
		return nil, fmt.Errorf("realmodel: forward: %w", err)
	}
	if cache != nil {
		cache.Sync()
	}
	return logits, nil
}

// Encode tokenizes a raw string into model token ids (no chat template).
func (m *Model) Encode(text string) ([]int32, error) {
	ids, err := m.Tok.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("realmodel: encode: %w", err)
	}
	return ids, nil
}

// systemPrompt nudges the base math model toward a short chain-of-thought that
// ends in a \boxed{} answer mathverify can extract.
const systemPrompt = "You are a helpful math assistant. Solve the problem and give the final answer inside \\boxed{}."

// EncodePrompt formats a math question through the model's chat template (with a
// math system prompt) and returns the prompt token ids ready for generation.
// Qwen2.5-Math is trained with this format; the raw question alone makes the
// base model ramble, so the smoke uses the template for coherent rollouts.
func (m *Model) EncodePrompt(question string) ([]int32, error) {
	msgs := []mlxlm.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: question},
	}
	ids, err := m.Tok.ApplyChatTemplate(msgs, true)
	if err != nil {
		// Fall back to a plain encode if the template is unavailable.
		return m.Encode(question)
	}
	return ids, nil
}

// Decode renders token ids back to text.
func (m *Model) Decode(ids []int32) (string, error) {
	s, err := m.Tok.Decode(ids)
	if err != nil {
		return "", fmt.Errorf("realmodel: decode: %w", err)
	}
	return s, nil
}

// EOS is the tokenizer's end-of-sequence token id.
func (m *Model) EOS() int32 { return m.Tok.EOSToken() }

// Close drops the model's references and clears the Metal buffer cache so a
// subsequent Load does not keep two multi-GB models resident (which OOMs). After
// Close the model must not be used. It does NOT manually free param arrays — the
// optimizer owns and frees the (possibly updated) weight arrays, so freeing them
// here would double-free; dropping the references lets GC + ClearCache reclaim
// them. The caller should runtime.GC() after Close to force finalization.
func (m *Model) Close() {
	m.LM = nil
	m.Tr = nil
	mlx.ClearCache()
}
