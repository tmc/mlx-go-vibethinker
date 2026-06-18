//go:build modelir

package realmodel

import (
	"context"
	"fmt"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"
)

// scored holds the per-token chosen-token log-probabilities of a rollout group
// under one policy, laid out as a [G, T] tensor with a matching [G, T] mask
// (1 for real completion tokens, 0 for padding). T is the longest completion in
// the group; shorter completions are right-padded and masked off.
//
// The per-token log-prob at completion position j is
//
//	logsoftmax(logits[promptLen-1+j])[completion[j]]
//
// i.e. the log-probability the policy assigns to the actually-generated token,
// read from the logits at the preceding position (next-token prediction).
type scored struct {
	logProbs *mlx.Array // [G, T], lazy (not yet evaluated) — carries gradient when from live weights
	mask     *mlx.Array // [G, T] float32, 1 on real tokens
	g, t     int
}

// rescoreGroup runs the model forward over each rollout's prompt+completion and
// gathers the per-token chosen-token log-probs into a [G, T] tensor plus mask.
//
// The returned logProbs is LAZY and shares the model's current parameter graph:
// when the model holds the LIVE (being-updated) weights this is the gradient-
// carrying `current`; to capture a frozen `old`/`ref` snapshot, wrap the result
// in mlx.StopGradient (see captureFrozen). The caller controls which weights the
// model holds before calling (SetWeights), so the same routine produces current,
// old, and ref depending on the model state at call time.
func rescoreGroup(ctx context.Context, lm models.LanguageModel, rollouts []rollout) (scored, error) {
	g := len(rollouts)
	if g == 0 {
		return scored{}, fmt.Errorf("realmodel: empty rollout group")
	}
	// Longest completion in the group sets T (padding target).
	t := 0
	for _, r := range rollouts {
		if len(r.completion) > t {
			t = len(r.completion)
		}
	}
	if t == 0 {
		return scored{}, fmt.Errorf("realmodel: all completions empty")
	}

	rows := make([]*mlx.Array, g) // each [1, T] log-probs
	maskVals := make([]float32, g*t)
	for i, r := range rollouts {
		full := r.full()
		promptLen := len(r.prompt)
		compLen := len(r.completion)

		in := mlx.NewArray(full, 1, len(full))
		logits, _ := lm.Forward(ctx, in, nil) // [1, len(full), vocab]; lazy
		// Per-token log-probs over the vocab: logsoftmax = logits - logsumexp.
		lse := mlx.LogsumexpAxis(logits, 2, true) // [1, len(full), 1]
		logProbsAll := mlx.Subtract(logits, lse)  // [1, len(full), vocab]

		// Gather the chosen-token log-prob at each completion position j from the
		// logits at position promptLen-1+j. Build a [1, T, 1] index of the chosen
		// token ids (padded positions index token 0 and are masked off).
		idxVals := make([]int32, t)
		for j := 0; j < t; j++ {
			if j < compLen {
				idxVals[j] = r.completion[j]
				maskVals[i*t+j] = 1
			} else {
				idxVals[j] = 0
				maskVals[i*t+j] = 0
			}
		}

		// Slice the log-prob rows aligned to completion positions: positions
		// [promptLen-1 .. promptLen-1+T-1] predict completion tokens [0..T-1].
		// Clamp the slice to the sequence length, then pad rows for any overrun.
		start := promptLen - 1
		stop := start + t
		seqLen := len(full)
		if stop > seqLen {
			stop = seqLen
		}
		predRows := mlx.Slice(logProbsAll,
			[]int{0, start, 0},
			[]int{1, stop, logitsVocab(logProbsAll)},
			[]int{1, 1, 1}) // [1, stop-start, vocab]

		idx := mlx.NewArray(idxVals[:stop-start], 1, stop-start, 1)
		gathered := mlx.TakeAlongAxis(predRows, idx, 2) // [1, stop-start, 1]
		row := mlx.Reshape(gathered, 1, stop-start)     // [1, stop-start]

		// Right-pad the row to T with zeros (masked off) if the sequence was
		// shorter than the padded T.
		if stop-start < t {
			padCols := t - (stop - start)
			pad := mlx.Zeros([]int{1, padCols}, row.Dtype())
			cat, err := mlx.ConcatenateAxis([]*mlx.Array{row, pad}, 1)
			if err != nil {
				return scored{}, fmt.Errorf("realmodel: pad logprob row: %w", err)
			}
			row = cat
		}
		rows[i] = row
	}

	logProbs, err := mlx.ConcatenateAxis(rows, 0) // [G, T]
	if err != nil {
		return scored{}, fmt.Errorf("realmodel: assemble logprobs: %w", err)
	}
	mask := mlx.NewArray(maskVals, g, t)
	return scored{logProbs: logProbs, mask: mask, g: g, t: t}, nil
}

// logitsVocab returns the vocab (last) dimension of a [..., vocab] array.
func logitsVocab(a *mlx.Array) int {
	s := a.Shape()
	return s[len(s)-1]
}

// captureFrozen evaluates and stop-gradients a scored group's log-probs so it
// can serve as a frozen `old` (behavior) or `ref` (reference) policy snapshot —
// a constant w.r.t. the gradient. The mask is shared (it is data, not a policy).
func captureFrozen(s scored) (scored, error) {
	frozen := mlx.StopGradient(s.logProbs)
	if err := mlx.Eval(frozen); err != nil {
		return scored{}, fmt.Errorf("realmodel: freeze logprobs: %w", err)
	}
	return scored{logProbs: frozen, mask: s.mask, g: s.g, t: s.t}, nil
}
