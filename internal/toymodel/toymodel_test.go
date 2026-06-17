package toymodel

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"
)

func TestBuildAndForward(t *testing.T) {
	lm, err := New(DefaultConfig(), 42)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Forward a short toy sequence; logits must be finite.
	tok := Tokenizer{}
	ids := tok.Encode("hello")
	in := make([]int32, len(ids))
	for i, id := range ids {
		in[i] = int32(id)
	}
	x := mlx.NewArray(in, 1, len(in))
	logits, _, err := models.Forward(context.Background(), lm, x, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := mlx.Eval(logits); err != nil {
		t.Fatalf("eval: %v", err)
	}
	shape := logits.Shape()
	if shape[0] != 1 || shape[len(shape)-1] != DefaultConfig().Vocab {
		t.Fatalf("logits shape = %v, want [1, seq, %d]", shape, DefaultConfig().Vocab)
	}
	vals, err := mlx.ToSlice[float32](logits)
	if err != nil {
		t.Fatalf("toslice: %v", err)
	}
	for _, v := range vals {
		if v != v { // NaN check
			t.Fatal("logits contain NaN")
		}
	}
}

// logitsOf runs a forward pass on a fixed toy sequence and returns the flat
// float32 logits.
func logitsOf(t *testing.T, lm models.LanguageModel) []float32 {
	t.Helper()
	x := mlx.NewArray([]int32{1, 5, 9, 13}, 1, 4)
	logits, _, err := models.Forward(context.Background(), lm, x, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if err := mlx.Eval(logits); err != nil {
		t.Fatalf("eval: %v", err)
	}
	v, err := mlx.ToSlice[float32](logits)
	if err != nil {
		t.Fatalf("toslice: %v", err)
	}
	return v
}

// TestWeightsRoundTrip saves a model, loads its weights into a differently
// seeded model, and confirms the loaded model reproduces the saved model's
// logits — the invariant a checkpoint round-trip must guarantee.
func TestWeightsRoundTrip(t *testing.T) {
	lm, err := New(DefaultConfig(), 7)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := logitsOf(t, lm)

	dir := t.TempDir()
	path := filepath.Join(dir, "toy.safetensors")
	if err := Save(lm, path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// A differently-seeded model has different logits...
	lm2, err := New(DefaultConfig(), 99)
	if err != nil {
		t.Fatalf("New 2: %v", err)
	}
	before := logitsOf(t, lm2)
	if maxAbsDiff(want, before) < 1e-3 {
		t.Fatal("control failed: different seeds produced identical logits")
	}

	// ...until we load the saved weights, after which logits must match.
	if err := LoadInto(lm2, path); err != nil {
		t.Fatalf("LoadInto: %v", err)
	}
	after := logitsOf(t, lm2)
	if d := maxAbsDiff(want, after); d > 1e-4 {
		t.Fatalf("round-trip logits differ by %v", d)
	}
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return 1e9
	}
	var m float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}
