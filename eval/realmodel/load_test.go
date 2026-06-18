//go:build modelir

package realmodel

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

// requireModel loads the real model, skipping the test when the weights are not
// present on the machine (the multi-GB download is not committed, so CI and
// other boxes legitimately lack it). When the model IS present the test runs for
// real — it does not pass vacuously.
func requireModel(t *testing.T) *Model {
	t.Helper()
	dir := DefaultModelDir()
	if dir == "" {
		t.Skip("realmodel: no home directory to locate the model")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("realmodel: model not present at %q (set %s or download Qwen2.5-Math-1.5B); skipping real-model smoke", dir, modelDirEnv)
	}
	m, err := Load(context.Background(), dir)
	if err != nil {
		t.Fatalf("realmodel: Load(%q): %v", dir, err)
	}
	return m
}

// A real load + single forward pass must produce finite logits whose last
// dimension is the tokenizer vocab and whose sequence dimension matches the
// prompt length — the guard that the real-model substrate wiring is sound before
// the per-method run sinks time into it.
func TestLoadAndForwardSmoke(t *testing.T) {
	m := requireModel(t)

	const prompt = "What is 2 + 2? The answer is"
	ids, err := m.Encode(prompt)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("encode produced no tokens")
	}

	logits, err := m.Forward(context.Background(), ids)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	defer logits.Free()

	// Shape: [1, seqLen, vocab].
	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits rank = %d, want 3 (got shape %v)", len(shape), shape)
	}
	if shape[0] != 1 {
		t.Errorf("logits batch dim = %d, want 1", shape[0])
	}
	if shape[1] != len(ids) {
		t.Errorf("logits seq dim = %d, want %d (prompt length)", shape[1], len(ids))
	}
	// The model's output vocab is padded up to a hardware-friendly size, so it is
	// >= the tokenizer's real vocab (every real token id has a logit).
	if shape[2] < m.Vocab() {
		t.Errorf("logits vocab dim = %d, want >= %d (tokenizer vocab)", shape[2], m.Vocab())
	}

	// Finiteness: the last position's logits must be all-finite (the real
	// generation/rescore path reads exactly this slice).
	last := lastPositionLogits(t, logits)
	if len(last) == 0 {
		t.Fatal("no logits read back")
	}
	var nonFinite int
	for _, v := range last {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nonFinite++
		}
	}
	if nonFinite > 0 {
		t.Fatalf("%d/%d last-position logits are non-finite", nonFinite, len(last))
	}

	// Sanity: the greedy (argmax) next token at the final position is a valid
	// vocab id — the same step the rollout path takes.
	argmax := mlx.ArgmaxAxis(logits, 2, false) // [1, seqLen]
	if err := mlx.Eval(argmax); err != nil {
		t.Fatalf("eval argmax: %v", err)
	}
	defer argmax.Free()
	// Argmax indices come back as uint32.
	greedy, err := mlx.ToSlice[uint32](argmax)
	if err != nil {
		t.Fatalf("argmax toslice: %v", err)
	}
	next := greedy[len(greedy)-1]
	if int(next) >= m.Vocab() {
		t.Fatalf("greedy next token %d out of vocab range [0,%d)", next, m.Vocab())
	}
	t.Logf("real-model smoke: prompt %d tokens, logits %v, greedy next id=%d", len(ids), shape, next)
}

// lastPositionLogits materializes the vocab-length logit row at the final
// sequence position as float32. logits is [1, seqLen, vocab]; the last row is
// the final vocab-length span of the flattened array.
func lastPositionLogits(t *testing.T, logits *mlx.Array) []float32 {
	t.Helper()
	shape := logits.Shape()
	vocab := shape[2]
	// Real-model logits are bf16; cast to float32 for the host-side read.
	f32, err := mlx.AsType[float32](logits)
	if err != nil {
		t.Fatalf("logits astype float32: %v", err)
	}
	defer f32.Free()
	all, err := mlx.ToSlice[float32](f32)
	if err != nil {
		t.Fatalf("logits toslice: %v", err)
	}
	if len(all) < vocab {
		t.Fatalf("logits flat len %d < vocab %d", len(all), vocab)
	}
	return all[len(all)-vocab:]
}
