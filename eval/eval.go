package eval

import (
	"context"
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// Params holds vLLM-equivalent sampling parameters for evaluation (DESIGN
// §4.7). TopK = -1 disables top-k truncation, matching vLLM's convention.
type Params struct {
	Temperature float64 // sampling temperature
	TopP        float64 // nucleus (top-p) cutoff
	TopK        int     // top-k cutoff; -1 disables
	MaxTokens   int     // maximum completion length in tokens
	K           int     // samples drawn per prompt for Pass@1
}

// MathParams are the paper's eval sampling params for math (and knowledge)
// benchmarks: temp 1.0, top_p 0.95, top_k -1, k=64.
var MathParams = Params{Temperature: 1.0, TopP: 0.95, TopK: -1, MaxTokens: 40000, K: 64}

// CodeParams are the paper's eval sampling params for code benchmarks:
// temp 0.6, top_p 0.95, top_k -1, k=8.
var CodeParams = Params{Temperature: 0.6, TopP: 0.95, TopK: -1, MaxTokens: 40000, K: 8}

// DefaultThreshold is the reward cutoff above which a sample counts as a
// success (reward ≥ threshold ⇒ 1). Verifiers that already emit {0,1} are
// unaffected.
const DefaultThreshold = 0.5

// A Sampler generates completions for a prompt. Sample returns n completions
// drawn under the given Params; it is the seam over a model decode loop and is
// stubbed by fixtures in tests. It must return exactly n completions or an
// error.
type Sampler interface {
	Sample(ctx context.Context, prompt string, p Params, n int) ([]string, error)
}

// SamplerFunc adapts a function to a Sampler.
type SamplerFunc func(ctx context.Context, prompt string, p Params, n int) ([]string, error)

// Sample implements Sampler.
func (f SamplerFunc) Sample(ctx context.Context, prompt string, p Params, n int) ([]string, error) {
	return f(ctx, prompt, p, n)
}

// Result reports a Pass@1 evaluation: the overall score, the per-prompt scores
// (in prompt order), and the number of samples drawn per prompt.
type Result struct {
	PassAt1   float64
	PerPrompt []float64
	K         int
}

// Pass1 evaluates Pass@1 of sampler over prompts, scoring each completion with
// verifier and counting a sample as a success when its reward is ≥ threshold.
// It draws p.K samples per prompt; p.K must be ≥ 1 and there must be at least
// one prompt. threshold ≤ 0 selects DefaultThreshold. The returned Result holds
// the overall Pass@1 and the per-prompt means.
func Pass1(ctx context.Context, sampler Sampler, verifier rl.Environment, prompts []string, p Params, threshold float64) (Result, error) {
	if sampler == nil {
		return Result{}, fmt.Errorf("eval: nil sampler")
	}
	if verifier == nil {
		return Result{}, fmt.Errorf("eval: nil verifier")
	}
	if len(prompts) == 0 {
		return Result{}, fmt.Errorf("eval: no prompts")
	}
	if p.K < 1 {
		return Result{}, fmt.Errorf("eval: k must be >= 1, got %d", p.K)
	}
	if threshold <= 0 {
		threshold = DefaultThreshold
	}
	rewards := make([][]float64, len(prompts))
	for i, q := range prompts {
		comps, err := sampler.Sample(ctx, q, p, p.K)
		if err != nil {
			return Result{}, fmt.Errorf("eval: sample prompt %d: %w", i, err)
		}
		if len(comps) != p.K {
			return Result{}, fmt.Errorf("eval: prompt %d: sampler returned %d completions, want %d", i, len(comps), p.K)
		}
		bin := make([]float64, p.K)
		for j, c := range comps {
			score, err := verifier.Score(ctx, q, c)
			if err != nil {
				return Result{}, fmt.Errorf("eval: score prompt %d sample %d: %w", i, j, err)
			}
			bin[j] = binary(score, threshold)
		}
		rewards[i] = bin
	}
	overall, per := pass1(rewards)
	return Result{PassAt1: overall, PerPrompt: per, K: p.K}, nil
}

// binary thresholds a (possibly non-binary) reward to {0,1}.
func binary(score, threshold float64) float64 {
	if score >= threshold {
		return 1
	}
	return 0
}

// pass1 is the numeric core: given per-prompt binary rewards (rewards[i][j] ∈
// {0,1}), it returns the overall Pass@1 and the per-prompt means. Pass@1 is the
// mean over prompts of the mean over samples:
//
//	Pass@1 = (1/|Q|) Σ_q (1/k) Σ_j r_j(q).
//
// rewards must be non-empty with non-empty rows. The per-prompt slice is in
// prompt order.
func pass1(rewards [][]float64) (overall float64, perPrompt []float64) {
	perPrompt = make([]float64, len(rewards))
	for i, row := range rewards {
		var s float64
		for _, r := range row {
			s += r
		}
		perPrompt[i] = s / float64(len(row))
	}
	var s float64
	for _, m := range perPrompt {
		s += m
	}
	return s / float64(len(perPrompt)), perPrompt
}
