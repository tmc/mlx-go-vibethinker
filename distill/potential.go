package distill

import (
	"context"
	"fmt"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// Score returns the learning-potential score S_LP for a trace given its
// per-token student log-probabilities (natural log). It is the length-
// normalized negative log-likelihood,
//
//	S_LP = −(1/n) Σ logProbs,
//
// where n = len(logProbs). A higher score means the student assigns lower
// probability to the trace — it is less well modeled and so more valuable to
// distill. An empty slice is an error (an undefined per-token average).
func Score(logProbs []float64) (float64, error) {
	if len(logProbs) == 0 {
		return 0, fmt.Errorf("distill: empty log-prob slice")
	}
	return score(logProbs), nil
}

// score is the unchecked core of Score.
func score(logProbs []float64) float64 {
	var sum float64
	for _, lp := range logProbs {
		sum += lp
	}
	return -sum / float64(len(logProbs))
}

// A StudentScorer produces the per-token log-probabilities the student assigns
// to a trace's tokens. It is the seam between the pure S_LP arithmetic and a
// real model forward; tests inject a fake.
type StudentScorer interface {
	// LogProbs returns the natural-log probability the student assigned to each
	// actual next token of the trace.
	LogProbs(ctx context.Context, tokens []int) ([]float64, error)
}

// modelScorer adapts an mlx-go-lm LanguageModel to StudentScorer via
// rl.LogProbs.
type modelScorer struct{ model models.LanguageModel }

// NewModelScorer returns a StudentScorer backed by a real model's forward pass.
func NewModelScorer(model models.LanguageModel) StudentScorer {
	return modelScorer{model: model}
}

func (m modelScorer) LogProbs(ctx context.Context, tokens []int) ([]float64, error) {
	if len(tokens) < 2 {
		return nil, fmt.Errorf("distill: need at least 2 tokens, got %d", len(tokens))
	}
	ids := make([]int32, len(tokens))
	for i, t := range tokens {
		ids[i] = int32(t)
	}
	arr := mlx.NewArray(ids, 1, len(ids))
	lp, err := rl.LogProbs(m.model, arr)
	if err != nil {
		return nil, fmt.Errorf("distill: student logprobs: %w", err)
	}
	if err := mlx.Eval(lp); err != nil {
		return nil, fmt.Errorf("distill: eval logprobs: %w", err)
	}
	out, err := mlx.ToSlice[float32](lp)
	if err != nil {
		return nil, fmt.Errorf("distill: read logprobs: %w", err)
	}
	res := make([]float64, len(out))
	for i, v := range out {
		res[i] = float64(v)
	}
	return res, nil
}

// ScoreTrace computes S_LP for a token trace using a StudentScorer.
func ScoreTrace(ctx context.Context, s StudentScorer, tokens []int) (float64, error) {
	lp, err := s.LogProbs(ctx, tokens)
	if err != nil {
		return 0, err
	}
	return Score(lp)
}
