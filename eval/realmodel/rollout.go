//go:build modelir

package realmodel

import (
	"context"
	"fmt"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"
)

// rollout is one generated completion for a prompt: the prompt token ids, the
// generated (completion) token ids, and the decoded completion text used for
// reward scoring.
type rollout struct {
	prompt     []int32 // prompt token ids
	completion []int32 // generated token ids (no prompt)
	text       string  // decoded completion text
}

// full returns the concatenated prompt+completion token ids — the sequence the
// rescore forward runs over.
func (r rollout) full() []int32 {
	out := make([]int32, 0, len(r.prompt)+len(r.completion))
	out = append(out, r.prompt...)
	out = append(out, r.completion...)
	return out
}

// generate samples one completion of up to maxTokens tokens from the model for a
// prompt, at the given temperature, using a fresh KV cache and a per-rollout
// random key for diversity. Sampling (not greedy) is what gives a group of K
// rollouts the within-group spread the group-relative advantage needs. It stops
// early on EOS.
func (m *Model) generate(ctx context.Context, prompt []int32, maxTokens int, temperature float64, key *mlx.Array) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("realmodel: empty prompt")
	}
	cache := models.NewKVCache()
	// Release the KV cache (and any reassigned successor) on every exit so caches
	// do not accumulate across the many rollouts of a run.
	defer func() { cache.Close() }()
	x := mlx.NewArray(prompt, 1, len(prompt))
	out := make([]int32, 0, maxTokens)
	eos := m.EOS()

	for step := 0; step < maxTokens; step++ {
		logits, next, err := models.Forward(ctx, m.LM, x, cache)
		if err != nil {
			x.Free()
			return out, fmt.Errorf("realmodel: generate forward: %w", err)
		}
		cache = next

		// Last-position logits, [1, vocab].
		shape := logits.Shape()
		lastIdx := shape[1] - 1
		last := mlx.Slice(logits,
			[]int{0, lastIdx, 0},
			[]int{1, lastIdx + 1, shape[2]},
			[]int{1, 1, 1}) // [1,1,vocab]
		lastR := mlx.Reshape(last, 1, shape[2]) // [1,vocab]

		// Sample at temperature: Categorical over logits/temperature. A distinct
		// subkey per step keeps the stream diverse across the completion.
		nextKey, sampleKey := random.Split(key, nil)
		key.Free()
		key = nextKey
		scaled := lastR
		if temperature > 0 && temperature != 1 {
			scaled = mlx.DivideScalar(lastR, float32(temperature))
		}
		tokArr := random.Categorical(scaled, -1, sampleKey, nil) // [1]
		if err := mlx.Eval(tokArr); err != nil {
			x.Free()
			return out, fmt.Errorf("realmodel: eval sample: %w", err)
		}
		cache.Sync()

		ids, err := mlx.ToSlice[uint32](tokArr)
		if err != nil {
			x.Free()
			return out, fmt.Errorf("realmodel: read sample: %w", err)
		}

		// Free this step's intermediates so the live-array count does not grow
		// with the completion length (long rollouts otherwise trip the array
		// resource limit). logits is owned by the cache graph — do not free it.
		// x is this step's input; free it before reassigning.
		last.Free()
		if scaled != lastR {
			scaled.Free()
		}
		lastR.Free()
		tokArr.Free()
		sampleKey.Free()
		x.Free()

		tok := int32(ids[0])
		out = append(out, tok)
		if tok == eos {
			x = nil
			break
		}
		x = mlx.NewArray([]int32{tok}, 1, 1)
	}
	if x != nil {
		x.Free()
	}
	key.Free()
	return out, nil
}

// rolloutGroup generates K sampled completions for one prompt and decodes each
// completion to text for reward scoring. The base key is split per rollout so
// the K completions are independent draws.
func (m *Model) rolloutGroup(ctx context.Context, p mathPrompt, k, maxTokens int, temperature float64, baseKey *mlx.Array) ([]rollout, error) {
	prompt, err := m.EncodePrompt(p.Question)
	if err != nil {
		return nil, err
	}
	rollouts := make([]rollout, 0, k)
	key := baseKey
	for i := 0; i < k; i++ {
		nextKey, rolloutKey := random.Split(key, nil)
		key = nextKey
		comp, err := m.generate(ctx, prompt, maxTokens, temperature, rolloutKey)
		if err != nil {
			return nil, err
		}
		text, err := m.Decode(comp)
		if err != nil {
			return nil, err
		}
		rollouts = append(rollouts, rollout{prompt: prompt, completion: comp, text: text})
		// Drain in-flight command buffers and release pooled Metal buffers between
		// rollouts so the live-resource count stays bounded over a long run
		// (otherwise the RoPE/forward kernels trip the device resource limit).
		mlx.Synchronize()
		mlx.ClearCache()
	}
	return rollouts, nil
}
