//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/reward/mathverify"
)

// HeldoutResult is the held-out correctness measurement at one policy state: the
// mean greedy Avg@1 over the fixed held-out set and the per-prompt 0/1 scores in
// set order. It is a DIRECTIONAL signal — a tiny FIXED probe set scored by greedy
// (argmax) decode against the boxed gold answer — NOT a benchmark. The same set
// is scored at step 0 and at the final step so the sweep ranks methods by the
// delta (final − step0).
type HeldoutResult struct {
	Acc    float64   // mean Avg@1 over the set
	Scores []float64 // per-prompt 0/1, in HeldoutSet order
	N      int       // |set|
}

// scoreHeldout greedily decodes a completion for each held-out prompt (argmax,
// no sampling — Avg@1) and scores it against the boxed gold answer with
// mathverify. It is deterministic given fixed weights (greedy decode), so the
// only run-to-run variation in a held-out delta comes from the optimizer steps
// between the two measurements, not from sampling noise. maxTokens bounds the
// decode; the math answers are short so a modest budget suffices.
//
// Greedy decode here is intentionally distinct from the temperature-sampled
// rollouts the GRPO loop trains on: the rollouts need within-group spread (so
// they sample), but the held-out CORRECTNESS signal must be a stable point
// estimate of the policy, so it decodes greedily.
func scoreHeldout(ctx context.Context, m *Model, set []mathPrompt, maxTokens int) (HeldoutResult, error) {
	if len(set) == 0 {
		return HeldoutResult{}, fmt.Errorf("realmodel: empty held-out set")
	}
	scores := make([]float64, len(set))
	var sum float64
	for i, p := range set {
		prompt, err := m.EncodePrompt(p.Question)
		if err != nil {
			return HeldoutResult{}, fmt.Errorf("realmodel: held-out encode %d: %w", i, err)
		}
		comp, err := m.generateGreedy(ctx, prompt, maxTokens)
		if err != nil {
			return HeldoutResult{}, fmt.Errorf("realmodel: held-out greedy decode %d: %w", i, err)
		}
		text, err := m.Decode(comp)
		if err != nil {
			return HeldoutResult{}, fmt.Errorf("realmodel: held-out decode %d: %w", i, err)
		}
		s := mathverify.Reward(text, p.Gold)
		scores[i] = s
		sum += s
		// Bound the live Metal-resource count across the held-out pass.
		reclaim()
	}
	return HeldoutResult{Acc: sum / float64(len(set)), Scores: scores, N: len(set)}, nil
}

// generateGreedy decodes a single completion by taking the argmax token at each
// step (no sampling), up to maxTokens, stopping at EOS or as soon as a complete
// \boxed{...} answer has been emitted. It is deterministic given the weights —
// the held-out Avg@1 must be a stable point estimate of the policy, not a sample.
//
// IMPORTANT — it re-runs the FULL forward over the growing sequence each step
// (no incremental KV cache). The substrate's incremental KV-cache decode
// (models.NewKVCache + models.Forward fed one token at a time, and equally the
// substrate's own generate.Greedy helper) produces a DEGENERATE REPETITION LOOP
// on this Qwen2.5-Math-1.5B — verified directly: the same prompt that the
// no-cache forward answers correctly ("…\boxed{11}") loops as "the number of the
// number of…" under the cache. So the held-out correctness signal cannot use the
// cached path; it uses the proven no-cache forward (the load smoke's path).
//
// The no-cache forward is O(n²) in the completion length and accumulates live
// Metal arrays, so two bounds keep it under the device array ceiling (~499000):
// the \boxed{} early-stop (the verbose CoT is cut the moment the answer lands,
// which is what mathverify scores anyway, typically well under 100 tokens), and
// a per-step free of the step's intermediates. The caller additionally reclaims
// between prompts.
func (m *Model) generateGreedy(ctx context.Context, prompt []int32, maxTokens int) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("realmodel: empty prompt")
	}
	eos := m.EOS()
	seq := append([]int32(nil), prompt...)
	out := make([]int32, 0, maxTokens)

	for step := 0; step < maxTokens; step++ {
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			return out, fmt.Errorf("realmodel: greedy forward: %w", err)
		}
		shape := logits.Shape()
		lastIdx := shape[1] - 1
		last := mlx.Slice(logits,
			[]int{0, lastIdx, 0},
			[]int{1, lastIdx + 1, shape[2]},
			[]int{1, 1, 1}) // [1,1,vocab]
		tok := mlx.ArgmaxAxis(last, 2, false) // [1,1]
		evalErr := mlx.Eval(tok)
		var ids []uint32
		if evalErr == nil {
			ids, evalErr = mlx.ToSlice[uint32](tok)
		}
		logits.Free()
		last.Free()
		tok.Free()
		if evalErr != nil {
			return out, fmt.Errorf("realmodel: greedy argmax: %w", evalErr)
		}

		t := int32(ids[0])
		if t == eos {
			break
		}
		out = append(out, t)
		seq = append(seq, t)

		// Early-stop once a complete \boxed{...} answer is on the page: the held-out
		// reward only reads that, and stopping bounds the O(n²) decode. Decoding
		// each step is cheap relative to the forward.
		if step%4 == 3 || step == maxTokens-1 {
			if text, derr := m.Decode(out); derr == nil && hasCompleteBoxed(text) {
				break
			}
		}
	}
	return out, nil
}

// hasCompleteBoxed reports whether text contains a brace-balanced \boxed{...}
// with a non-empty body — the held-out early-stop signal that the answer
// mathverify will read is fully on the page.
func hasCompleteBoxed(text string) bool {
	i := strings.LastIndex(text, "\\boxed{")
	if i < 0 {
		return false
	}
	depth := 0
	body := false
	for j := i + len("\\boxed{") - 1; j < len(text); j++ {
		switch text[j] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return body
			}
		default:
			if depth == 1 && text[j] != ' ' {
				body = true
			}
		}
	}
	return false
}
