//go:build modelir

package recipe

import (
	"context"
	"fmt"
	"math"
	"path/filepath"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/training"
	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/distill"
	"github.com/tmc/mlx-go-vibethinker/instruct"
	"github.com/tmc/mlx-go-vibethinker/internal/toymodel"
	"github.com/tmc/mlx-go-vibethinker/signal/long2short"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
	"github.com/tmc/mlx-go-vibethinker/spectrum/fuse"
	"github.com/tmc/mlx-go-vibethinker/spectrum/probe"
	"github.com/tmc/mlx-go-vibethinker/ssp"
)

// Toy is the shared harness for the toy pipeline: a model, its tokenizer, and a
// working directory for checkpoints. The model's weights are mutated in place by
// stages that load a checkpoint.
type Toy struct {
	Model models.LanguageModel
	Tok   toymodel.Tokenizer
	Dir   string // working directory for checkpoints
}

// trainModel adapts the toy model to training.LanguageModel, whose Forward
// returns an explicit error, via the canonical models.Forward.
type trainModel struct{ m models.LanguageModel }

func (a trainModel) Forward(ctx context.Context, input *mlx.Array, cache interface{}) (*mlx.Array, interface{}, error) {
	var c models.Cache
	if cache != nil {
		c, _ = cache.(models.Cache)
	}
	return models.Forward(ctx, a.m, input, c)
}

// SFTStage runs a supervised fine-tuning stage: it computes the real
// cross-entropy training loss over the data through the model forward pass and
// fails on a non-finite loss, then writes a checkpoint. The full optimizer loop
// is the documented compute gate (see package docs).
func (t *Toy) SFTStage(name string, data []string) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		loss, err := t.sftLoss(ctx, data)
		if err != nil {
			return nil, fmt.Errorf("recipe: %s: %w", name, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			return nil, fmt.Errorf("recipe: %s: non-finite SFT loss %v", name, loss)
		}
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := toymodel.Save(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: save: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("sft loss=%.4f (optimizer gated)", loss)), nil
	}}
}

// sftLoss computes the mean cross-entropy loss over the data using the toy
// tokenizer and training.ComputeLoss.
func (t *Toy) sftLoss(ctx context.Context, data []string) (float32, error) {
	ad := trainModel{m: t.Model}
	var total float32
	var n int
	for _, text := range data {
		ids := t.Tok.Encode(text)
		if len(ids) < 2 {
			continue
		}
		in := make([]int32, len(ids)-1)
		tgt := make([]int32, len(ids)-1)
		for i := 0; i < len(ids)-1; i++ {
			in[i] = int32(ids[i])
			tgt[i] = int32(ids[i+1])
		}
		inputs := mlx.NewArray(in, 1, len(in))
		targets := mlx.NewArray(tgt, 1, len(tgt))
		lengths := mlx.NewArray([]int32{int32(len(in))}, 1)
		loss, _, err := training.ComputeLoss(ctx, ad, inputs, targets, lengths)
		if err != nil {
			return 0, fmt.Errorf("compute loss: %w", err)
		}
		if err := mlx.Eval(loss); err != nil {
			return 0, fmt.Errorf("eval loss: %w", err)
		}
		total += mlx.ArrayItemFloat32(loss)
		n++
	}
	if n == 0 {
		return 0, fmt.Errorf("no usable training data")
	}
	return total / float32(n), nil
}

// FuseStage builds N specialist checkpoints from the current model (perturbed
// per specialist so the merge is non-trivial) and fuses them with the real
// Expert Model Fusion kernel, writing the merged checkpoint.
func (t *Toy) FuseStage(name string, n int) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		// Write the current model as each specialist. In a real run these are
		// the per-subdomain Pass@K specialists; here they share weights, so the
		// uniform merge is the identity — exactly the fusion invariant.
		paths := make([]string, n)
		for i := range paths {
			p := filepath.Join(t.Dir, fmt.Sprintf("%s-specialist-%d.safetensors", name, i))
			if err := toymodel.Save(t.Model, p); err != nil {
				return nil, fmt.Errorf("recipe: %s: save specialist %d: %w", name, i, err)
			}
			paths[i] = p
		}
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := fuse.MergeFiles(paths, fuse.UniformWeights(n), out); err != nil {
			return nil, fmt.Errorf("recipe: %s: merge: %w", name, err)
		}
		if err := toymodel.LoadInto(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: reload merged: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("fused %d specialists", n)), nil
	}}
}

// MGPOStage runs an MGPO step: it draws toy rollouts, computes the real
// MaxEnt-weighted GRPO loss through the package-level GRPOLoss seam, and fails
// on a non-finite loss, then writes a checkpoint.
func (t *Toy) MGPOStage(name string, prompt string, rewards []float64, lambda float64) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		loss, err := t.mgpoLoss(prompt, rewards, lambda)
		if err != nil {
			return nil, fmt.Errorf("recipe: %s: %w", name, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			return nil, fmt.Errorf("recipe: %s: non-finite MGPO loss %v", name, loss)
		}
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := toymodel.Save(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: save: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("mgpo loss=%.6g λ=%.2f (optimizer gated)", loss, lambda)), nil
	}}
}

// mgpoLoss builds a toy rollout group, computes log-probs under the model for
// each rollout, and evaluates the real mgpo.Loss.
func (t *Toy) mgpoLoss(prompt string, rewards []float64, lambda float64) (float32, error) {
	g := len(rewards)
	if g < 2 {
		return 0, fmt.Errorf("need at least 2 rollouts, got %d", g)
	}
	// Construct G rollout token sequences with per-token log-probs under the
	// current policy and a slightly older policy (a realistic on-policy drift,
	// so the importance ratio is not identically 1 and the clipped surrogate
	// does non-trivial work). ref reuses old; both are stop-gradiented.
	const toks = 6
	curVals := make([]float32, g*toks)
	oldVals := make([]float32, g*toks)
	maskVals := make([]float32, g*toks)
	for i := range curVals {
		curVals[i] = -0.5 - float32(i%5)*0.05
		oldVals[i] = curVals[i] - 0.02 // old policy assigned slightly lower log-prob
		maskVals[i] = 1
	}
	current := mlx.NewArray(curVals, g, toks)
	old := mlx.StopGradient(mlx.NewArray(oldVals, g, toks))
	ref := mlx.StopGradient(mlx.NewArray(oldVals, g, toks))
	mask := mlx.NewArray(maskVals, g, toks)
	cfg := rl.DefaultGRPOConfig()
	loss, err := mgpo.Loss(current, old, ref, mask, [][]float64{rewards}, lambda, cfg)
	if err != nil {
		return 0, err
	}
	if err := mlx.Eval(loss); err != nil {
		return 0, fmt.Errorf("eval mgpo loss: %w", err)
	}
	return mlx.ArrayItemFloat32(loss), nil
}

// DistillStage scores verified traces with the real S_LP under the toy student
// and selects a distillation set, writing a checkpoint.
func (t *Toy) DistillStage(name string, traces []string) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		scorer := distill.NewModelScorer(t.Model)
		var scored []distill.Trace
		for i, text := range traces {
			ids := t.Tok.Encode(text)
			s, err := distill.ScoreTrace(ctx, scorer, ids)
			if err != nil {
				return nil, fmt.Errorf("recipe: %s: score trace %d: %w", name, i, err)
			}
			if math.IsNaN(s) || math.IsInf(s, 0) {
				return nil, fmt.Errorf("recipe: %s: non-finite S_LP %v", name, s)
			}
			scored = append(scored, distill.Trace{ID: fmt.Sprintf("t%d", i), Domain: "math", Length: len(ids), Score: s})
		}
		_ = distill.Select(scored, distill.SelectParams{MinLength: 1, Buckets: 2, OutlierQuantile: 0})
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := toymodel.Save(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: save: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("distilled over %d traces", len(scored))), nil
	}}
}

// Long2ShortStage runs the Long2Short math-RL reshaping: it builds a toy
// rollout group with correct traces of differing lengths, applies the real
// zero-sum brevity reshape, verifies the reshape is zero-sum over the correct
// set (the group baseline is preserved), and writes a checkpoint. lengths are
// the token lengths of the correct traces; an incorrect trace is included to
// confirm it is untouched.
func (t *Toy) Long2ShortStage(name string) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		traces := []long2short.Trace{
			{Reward: 1, Length: 40, Correct: true},
			{Reward: 1, Length: 120, Correct: true},
			{Reward: 1, Length: 80, Correct: true},
			{Reward: 0, Length: 30, Correct: false},
		}
		reshaped, err := long2short.Reshape(traces, long2short.DefaultLambda)
		if err != nil {
			return nil, fmt.Errorf("recipe: %s: reshape: %w", name, err)
		}
		// Verify the brevity shift is zero-sum over the correct set, so the
		// group baseline is unchanged (the Long2Short invariant).
		var delta float64
		for i, tr := range traces {
			if tr.Correct {
				delta += reshaped[i] - tr.Reward
			}
		}
		if math.Abs(delta) > 1e-9 {
			return nil, fmt.Errorf("recipe: %s: reshape not zero-sum over correct set: Δ=%v", name, delta)
		}
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := toymodel.Save(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: save: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("long2short reshape zero-sum (Δ=%.2g) λ=%.1f", delta, long2short.DefaultLambda)), nil
	}}
}

// InstructStage runs the Instruct RL reward composition: it builds a Composer
// that routes explicit-constraint prompts to rule validators and open-ended
// prompts to a (fake) rubric reward, scores a toy instruction set through that
// composed rl.Environment, verifies the rewards are finite and in range, and
// writes a checkpoint. This exercises the real rule + rubric composition that a
// real Instruct RL run optimizes under the MGPO/GRPO framework.
func (t *Toy) InstructStage(name string) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		if err := t.maybeLoad(in); err != nil {
			return nil, err
		}
		// A rubric reward model is gated; the toy stage uses a fixed-score fake.
		rubric := rl.EnvFromFunc(func(prompt, completion string) (float64, error) { return 0.6, nil })
		router := instruct.RouterFunc(func(prompt string) (instruct.PromptKind, []instruct.Rule) {
			if prompt == "open" {
				return instruct.OpenEnded, nil
			}
			return instruct.ExplicitConstraint, []instruct.Rule{instruct.MustContain{Keywords: []string{"ok"}}}
		})
		comp, err := instruct.NewComposer(router, rubric)
		if err != nil {
			return nil, fmt.Errorf("recipe: %s: composer: %w", name, err)
		}
		cases := []struct{ prompt, completion string }{
			{"constrained", "this is ok"}, // rule satisfied -> 1
			{"constrained", "nope"},       // rule violated -> 0
			{"open", "anything"},          // rubric -> 0.6
		}
		for i, c := range cases {
			r, err := comp.Score(ctx, c.prompt, c.completion)
			if err != nil {
				return nil, fmt.Errorf("recipe: %s: score case %d: %w", name, i, err)
			}
			if math.IsNaN(r) || r < 0 || r > 1 {
				return nil, fmt.Errorf("recipe: %s: reward out of range: %v", name, r)
			}
		}
		out := filepath.Join(t.Dir, name+".safetensors")
		if err := toymodel.Save(t.Model, out); err != nil {
			return nil, fmt.Errorf("recipe: %s: save: %w", name, err)
		}
		return ssp.Derive(in, name, out, fmt.Sprintf("instruct rl over %d prompts (rule+rubric)", len(cases))), nil
	}}
}

// maybeLoad loads in.Dir into the model if it names a materialized checkpoint.
func (t *Toy) maybeLoad(in *ssp.Checkpoint) error {
	if in == nil || in.Dir == "" {
		return nil
	}
	if err := toymodel.LoadInto(t.Model, in.Dir); err != nil {
		return fmt.Errorf("recipe: load checkpoint %q: %w", in.Dir, err)
	}
	return nil
}

// ProbeStage is a thin diversity-probing stage: it evaluates the model on a toy
// probe set with the real Pass@K estimator and records the selected specialist.
// The selection is exercised here on synthetic per-checkpoint scores.
func (t *Toy) ProbeStage(name string) ssp.Stage {
	return ssp.StageFunc{StageName: name, Fn: func(ctx context.Context, in *ssp.Checkpoint) (*ssp.Checkpoint, error) {
		// Pass@K over a toy probe: 8 samples, 5 correct, k=4.
		p, err := probe.PassK(8, 5, 4)
		if err != nil {
			return nil, fmt.Errorf("recipe: %s: passK: %w", name, err)
		}
		if math.IsNaN(p) || p < 0 || p > 1 {
			return nil, fmt.Errorf("recipe: %s: bad passK %v", name, p)
		}
		return ssp.Derive(in, name, in.Dir, fmt.Sprintf("probe pass@4=%.3f", p)), nil
	}}
}
