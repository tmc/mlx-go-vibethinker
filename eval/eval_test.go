package eval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// constSampler returns the same n completions for every prompt; the content is
// just an index marker the fake verifier keys on.
type constSampler struct{}

func (constSampler) Sample(_ context.Context, _ string, _ Params, n int) ([]string, error) {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("sample-%d", i)
	}
	return out, nil
}

// allCorrect scores every completion as correct; allWrong as wrong.
func allCorrect() rl.Environment {
	return rl.EnvFromFunc(func(_, _ string) (float64, error) { return 1, nil })
}
func allWrong() rl.Environment {
	return rl.EnvFromFunc(func(_, _ string) (float64, error) { return 0, nil })
}

// fractionCorrect marks the first ceil(frac*K) of K samples correct, by parsing
// the index encoded in the completion string. With K samples and frac=0.5 the
// per-prompt mean is exactly 0.5.
func fractionCorrect(k int, frac float64) rl.Environment {
	cut := int(math.Round(frac * float64(k)))
	return rl.EnvFromFunc(func(_, completion string) (float64, error) {
		var idx int
		if _, err := fmt.Sscanf(completion, "sample-%d", &idx); err != nil {
			return 0, err
		}
		if idx < cut {
			return 1, nil
		}
		return 0, nil
	})
}

// Property (task): Pass@1 of an all-correct sampler is 1.
func TestPass1AllCorrect(t *testing.T) {
	p := Params{K: 8, TopK: -1}
	got, err := Pass1(context.Background(), constSampler{}, allCorrect(),
		[]string{"a", "b", "c"}, p, 0)
	if err != nil {
		t.Fatalf("Pass1: %v", err)
	}
	if got.PassAt1 != 1 {
		t.Fatalf("all-correct Pass@1 = %v, want 1", got.PassAt1)
	}
	for i, m := range got.PerPrompt {
		if m != 1 {
			t.Fatalf("prompt %d per-prompt = %v, want 1", i, m)
		}
	}
}

// Property (task): Pass@1 of a half-correct sampler is 0.5.
func TestPass1HalfCorrect(t *testing.T) {
	const k = 8
	p := Params{K: k, TopK: -1}
	got, err := Pass1(context.Background(), constSampler{}, fractionCorrect(k, 0.5),
		[]string{"a", "b", "c", "d"}, p, 0)
	if err != nil {
		t.Fatalf("Pass1: %v", err)
	}
	if math.Abs(got.PassAt1-0.5) > 1e-12 {
		t.Fatalf("half-correct Pass@1 = %v, want 0.5", got.PassAt1)
	}
}

func TestPass1AllWrong(t *testing.T) {
	p := Params{K: 4, TopK: -1}
	got, err := Pass1(context.Background(), constSampler{}, allWrong(),
		[]string{"a", "b"}, p, 0)
	if err != nil {
		t.Fatalf("Pass1: %v", err)
	}
	if got.PassAt1 != 0 {
		t.Fatalf("all-wrong Pass@1 = %v, want 0", got.PassAt1)
	}
}

// pass1 is the unbiased mean-of-means; check the core directly on a table where
// prompts have different per-prompt means.
func TestPass1Core(t *testing.T) {
	tests := []struct {
		name    string
		rewards [][]float64
		want    float64
		wantPer []float64
	}{
		{"all-one", [][]float64{{1, 1}, {1, 1}}, 1, []float64{1, 1}},
		{"all-zero", [][]float64{{0, 0}, {0, 0}}, 0, []float64{0, 0}},
		{"mixed", [][]float64{{1, 0, 0, 0}, {1, 1, 1, 1}}, 0.625, []float64{0.25, 1}},
		{"single-prompt", [][]float64{{1, 0}}, 0.5, []float64{0.5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, per := pass1(tt.rewards)
			if math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("pass1 = %v, want %v", got, tt.want)
			}
			for i := range per {
				if math.Abs(per[i]-tt.wantPer[i]) > 1e-12 {
					t.Fatalf("perPrompt[%d] = %v, want %v", i, per[i], tt.wantPer[i])
				}
			}
		})
	}
}

// Non-binary verifier rewards are thresholded to {0,1}; a reward exactly at the
// threshold counts as a success.
func TestThresholding(t *testing.T) {
	const k = 4
	// Scores 0.4 and 0.6 around the default 0.5 threshold; 0.5 itself passes.
	env := rl.EnvFromFunc(func(_, completion string) (float64, error) {
		var idx int
		fmt.Sscanf(completion, "sample-%d", &idx)
		switch idx {
		case 0:
			return 0.4, nil // below -> 0
		case 1:
			return 0.5, nil // at threshold -> 1
		default:
			return 0.6, nil // above -> 1
		}
	})
	p := Params{K: k, TopK: -1}
	got, err := Pass1(context.Background(), constSampler{}, env, []string{"q"}, p, 0)
	if err != nil {
		t.Fatalf("Pass1: %v", err)
	}
	// 3 of 4 samples pass.
	if math.Abs(got.PassAt1-0.75) > 1e-12 {
		t.Fatalf("Pass@1 = %v, want 0.75", got.PassAt1)
	}
}

// A custom threshold reclassifies borderline rewards.
func TestCustomThreshold(t *testing.T) {
	env := rl.EnvFromFunc(func(_, _ string) (float64, error) { return 0.6, nil })
	p := Params{K: 2, TopK: -1}
	// threshold 0.7 -> 0.6 fails -> Pass@1 = 0.
	got, err := Pass1(context.Background(), constSampler{}, env, []string{"q"}, p, 0.7)
	if err != nil {
		t.Fatalf("Pass1: %v", err)
	}
	if got.PassAt1 != 0 {
		t.Fatalf("Pass@1 = %v, want 0 under threshold 0.7", got.PassAt1)
	}
}

func TestParamsDefaults(t *testing.T) {
	if MathParams.Temperature != 1.0 || MathParams.TopP != 0.95 || MathParams.TopK != -1 || MathParams.K != 64 {
		t.Fatalf("MathParams = %+v, want temp 1.0 top_p 0.95 top_k -1 k 64", MathParams)
	}
	if CodeParams.Temperature != 0.6 || CodeParams.TopP != 0.95 || CodeParams.TopK != -1 || CodeParams.K != 8 {
		t.Fatalf("CodeParams = %+v, want temp 0.6 top_p 0.95 top_k -1 k 8", CodeParams)
	}
}

func TestPass1Validation(t *testing.T) {
	ctx := context.Background()
	good := Params{K: 2, TopK: -1}
	if _, err := Pass1(ctx, nil, allCorrect(), []string{"q"}, good, 0); err == nil {
		t.Fatal("want error for nil sampler")
	}
	if _, err := Pass1(ctx, constSampler{}, nil, []string{"q"}, good, 0); err == nil {
		t.Fatal("want error for nil verifier")
	}
	if _, err := Pass1(ctx, constSampler{}, allCorrect(), nil, good, 0); err == nil {
		t.Fatal("want error for no prompts")
	}
	if _, err := Pass1(ctx, constSampler{}, allCorrect(), []string{"q"}, Params{K: 0}, 0); err == nil {
		t.Fatal("want error for k < 1")
	}
}

// A sampler returning the wrong count is a hard error, not a silent miscount.
func TestSamplerCountMismatch(t *testing.T) {
	bad := SamplerFunc(func(_ context.Context, _ string, _ Params, n int) ([]string, error) {
		return []string{"only-one"}, nil // returns 1 regardless of n
	})
	_, err := Pass1(context.Background(), bad, allCorrect(), []string{"q"}, Params{K: 4, TopK: -1}, 0)
	if err == nil {
		t.Fatal("want error for completion count mismatch")
	}
}

func TestSamplerError(t *testing.T) {
	sentinel := errors.New("boom")
	bad := SamplerFunc(func(_ context.Context, _ string, _ Params, _ int) ([]string, error) {
		return nil, sentinel
	})
	_, err := Pass1(context.Background(), bad, allCorrect(), []string{"q"}, Params{K: 2, TopK: -1}, 0)
	if !errors.Is(err, sentinel) {
		t.Fatalf("error = %v, want wrapped %v", err, sentinel)
	}
}
