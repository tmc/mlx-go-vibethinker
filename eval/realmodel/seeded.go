//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"strconv"
)

// buildSeededGroups constructs rollout groups from FIXED completions rather than
// model generations, guaranteeing genuine within-group reward spread (some
// completions correct, some wrong) so the reward-shape mechanisms — Dr.GRPO's
// std removal, DCPO's cross-step smoothing, Dynamic Sampling's acc∈(0,1) keep —
// are observable on real logits.
//
// IMPORTANT: only the completion TEXT is fixed. Each completion is tokenized by
// the real tokenizer and rescored through the real model's forward pass, and the
// loss + optimizer step are fully real. The seeded block is labeled as such in
// the table/JSON and must NOT be read as model accuracy — it exhibits the
// mechanism, not the model's solving ability.
//
// For each prompt the group is built with a deterministic correct/incorrect mix
// keyed off the group index so the accuracy is strictly between 0 and 1 (a real
// learnable group), and one prompt is left all-wrong to keep a real cliff group
// in the seeded set too.
func buildSeededGroups(ctx context.Context, m *Model, cfg Config) ([]group, error) {
	prompts := mathPrompts()
	if cfg.Prompts < len(prompts) {
		prompts = prompts[:cfg.Prompts]
	}
	k := cfg.K
	if k < 4 {
		k = 4 // keep K>=4 so within-group spread is real
	}

	groups := make([]group, 0, len(prompts))
	for gi, p := range prompts {
		rollouts := make([]rollout, 0, k)
		prompt, err := m.EncodePrompt(p.Question)
		if err != nil {
			return nil, err
		}
		// Correct count per group: a rotating 1..k-1 so accuracy is in (0,1),
		// except the last prompt which is left all-wrong (a seeded cliff group).
		nCorrect := 1 + (gi % (k - 1))
		if gi == len(prompts)-1 {
			nCorrect = 0
		}
		for j := 0; j < k; j++ {
			var text string
			if j < nCorrect {
				text = seededCorrect(p.Gold)
			} else {
				text = seededWrong(p.Gold, j)
			}
			ids, err := m.Encode(text)
			if err != nil {
				return nil, err
			}
			if len(ids) == 0 {
				return nil, fmt.Errorf("realmodel: seeded completion tokenized empty")
			}
			rollouts = append(rollouts, rollout{prompt: prompt, completion: ids, text: text})
		}
		g, err := scoreAndFreeze(ctx, m, p, gi, rollouts)
		if err != nil {
			return nil, err
		}
		groups = append(groups, g)
		reclaim()
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("realmodel: no seeded groups")
	}
	return groups, nil
}

// seededCorrect renders a short, plausible correct completion ending in the gold
// boxed answer (which mathverify.Reward scores as 1).
func seededCorrect(gold string) string {
	return "Computing the value step by step gives the result. The final answer is \\boxed{" + gold + "}."
}

// seededWrong renders a wrong completion: the gold answer perturbed by a small
// offset so mathverify scores it 0, varied by index for within-group diversity
// (so DRA has a non-degenerate similarity structure too).
func seededWrong(gold string, idx int) string {
	wrong := gold
	if n, err := strconv.Atoi(gold); err == nil {
		wrong = strconv.Itoa(n + 1 + idx) // off by 1+idx -> definitely wrong
	} else {
		wrong = gold + "0"
	}
	return "Working through the problem, the result appears to be \\boxed{" + wrong + "}."
}
