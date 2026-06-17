package mgpo

import (
	"fmt"
	"math"
)

// HDPO privileged self-distillation on "cliff" groups (arXiv 2603.23871). On a
// zero-reward group — every rollout for a prompt fails the verifier — the
// group-relative advantage is identically zero and GRPO produces no gradient: a
// "gradient cliff" on exactly the hardest prompts. HDPO recovers a learning
// signal there without any external model: it conditions the SAME policy on the
// gold answer y* to draw privileged rollouts ȳ ~ π_θ(·|x⊕y*), keeps the correct
// ones, and distills the policy toward its own privileged distribution with a
// Jensen-Shannon term added only on the cliff set C = {x : Σ_k R = 0}:
//
//	L_HDPO = L_GRPO + λ_JSD · L_JSD,   L_JSD over C only.
//
// This file holds the self-contained, model-free core: identifying the cliff
// set, and the JSD between the policy and its privileged (teacher) distribution.
// The privileged-rollout generation — a forward pass of the same weights
// conditioned on y* — needs the model and lives behind the modelir build tag in
// the recipe/harness. With λ_JSD = 0 the JSD term contributes nothing and the
// objective is exactly L_GRPO, so the off-path is the unmodified baseline.

// HDPOConfig configures the HDPO cliff-JSD term. Its zero value (LambdaJSD = 0)
// adds nothing: HDPO degenerates to plain GRPO on every group, cliff or not.
type HDPOConfig struct {
	// LambdaJSD weights the cliff-set JSD distillation term. 0 disables HDPO
	// (baseline GRPO). The paper uses a small positive value; the term is added
	// to the loss only for groups in the cliff set.
	LambdaJSD float64
}

// IsCliffGroup reports whether a reward group is on the gradient cliff: every
// rollout scored zero (Σ_k R = 0, treating any reward > 0 as a success). Such a
// group has zero group variance and a zero group-relative advantage, so GRPO
// produces no gradient for it. An empty group is not a cliff group.
func IsCliffGroup(group []float64) bool {
	if len(group) == 0 {
		return false
	}
	return Accuracy(group) == 0
}

// CliffSet returns the indices of the reward groups on the gradient cliff
// (every rollout failed). These are exactly the groups where HDPO adds its
// privileged-self-distillation term and where plain GRPO learns nothing.
func CliffSet(rewards [][]float64) []int {
	idx := make([]int, 0, len(rewards))
	for i, g := range rewards {
		if IsCliffGroup(g) {
			idx = append(idx, i)
		}
	}
	return idx
}

// JSD returns the Jensen-Shannon divergence between two discrete distributions p
// and q (same length), in nats:
//
//	M = (p+q)/2,   JSD(p‖q) = ½·KL(p‖M) + ½·KL(q‖M).
//
// JSD is symmetric, non-negative, and bounded by ln 2. p and q must be the same
// length, non-negative, and each sum to a positive value (they are normalized
// internally, so unnormalized top-k weight vectors are accepted). A zero
// component contributes nothing (0·log0 ≡ 0). This is the per-rollout teacher
// term L_JSD distills: p is the policy distribution over the privileged tokens,
// q is the same policy conditioned on y* (the self-teacher).
func JSD(p, q []float64) (float64, error) {
	if len(p) != len(q) {
		return 0, fmt.Errorf("mgpo: JSD length mismatch %d != %d", len(p), len(q))
	}
	if len(p) == 0 {
		return 0, fmt.Errorf("mgpo: JSD on empty distributions")
	}
	pn, err := normalize(p, "p")
	if err != nil {
		return 0, err
	}
	qn, err := normalize(q, "q")
	if err != nil {
		return 0, err
	}
	var jsd float64
	for i := range pn {
		m := 0.5 * (pn[i] + qn[i])
		if pn[i] > 0 {
			jsd += 0.5 * pn[i] * math.Log(pn[i]/m)
		}
		if qn[i] > 0 {
			jsd += 0.5 * qn[i] * math.Log(qn[i]/m)
		}
	}
	if jsd < 0 {
		jsd = 0 // guard floating-point error; JSD ≥ 0 by Jensen.
	}
	return jsd, nil
}

// HDPOLossTerm returns the per-group HDPO objective contribution:
// baseGRPOLoss + λ_JSD·L_JSD, where L_JSD is the mean JSD over the cliff set and
// zero on non-cliff groups. baseGRPOLoss is the scalar GRPO loss already
// computed for the batch. cliffJSD[i] is the JSD between the policy and its
// privileged distribution for group i (only consulted for groups in the cliff
// set, identified from rewards); pass nil or any value for non-cliff groups.
//
// With cfg.LambdaJSD = 0 the result equals baseGRPOLoss exactly, so the off-path
// is bit-identical. The JSD term is added to the loss (not the advantage and not
// the reward), so the MGPO no-op rule is untouched: HDPO is a loss-side addition
// gated entirely on the cliff set.
func HDPOLossTerm(baseGRPOLoss float64, rewards [][]float64, cliffJSD []float64, cfg HDPOConfig) (float64, error) {
	if cfg.LambdaJSD == 0 {
		return baseGRPOLoss, nil
	}
	cliff := CliffSet(rewards)
	if len(cliff) == 0 {
		return baseGRPOLoss, nil // no cliff groups: HDPO adds nothing this step.
	}
	if len(cliffJSD) != len(rewards) {
		return 0, fmt.Errorf("mgpo: HDPO needs one JSD per group: %d != %d", len(cliffJSD), len(rewards))
	}
	var sum float64
	for _, i := range cliff {
		jsd := cliffJSD[i]
		if jsd < 0 || math.IsNaN(jsd) || math.IsInf(jsd, 0) {
			return 0, fmt.Errorf("mgpo: HDPO group %d has invalid JSD %v", i, jsd)
		}
		sum += jsd
	}
	meanJSD := sum / float64(len(cliff))
	return baseGRPOLoss + cfg.LambdaJSD*meanJSD, nil
}

// normalize returns p scaled to sum to 1, erroring on a non-positive or invalid
// total or any negative component.
func normalize(p []float64, name string) ([]float64, error) {
	var sum float64
	for _, x := range p {
		if x < 0 || math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("mgpo: JSD %s has invalid component %v", name, x)
		}
		sum += x
	}
	if sum <= 0 {
		return nil, fmt.Errorf("mgpo: JSD %s sums to %v, want > 0", name, sum)
	}
	out := make([]float64, len(p))
	for i, x := range p {
		out[i] = x / sum
	}
	return out, nil
}
