package mgpo

import (
	"fmt"
	"math"
)

// pClamp bounds p_c away from 0 and 1 so the logs in the Bernoulli KL stay
// finite. The clamp is applied symmetrically; with G rollouts the smallest
// nonzero deviation from an extreme is 1/G, so pClamp must be well below any
// realistic 1/G.
const pClamp = 1e-9

// p0 is the max-entropy reference accuracy: the Bernoulli distribution of
// maximum entropy has success probability 1/2.
const p0 = 0.5

// DME returns the max-entropy deviation D_ME(p_c ‖ 1/2): the Bernoulli KL
// divergence of an accuracy p_c from the maximum-entropy reference p₀ = 1/2,
//
//	D_ME = p_c·log(p_c/p₀) + (1−p_c)·log((1−p_c)/(1−p₀)).
//
// It is the additive (sum-of-two-terms) Bernoulli KL — the 1.5B PDF renders a
// stray "*" between the terms, but the 3B paper and the standard KL confirm the
// additive form. D_ME ≥ 0, with D_ME = 0 exactly at p_c = 1/2, and it grows
// toward log 2 as p_c approaches 0 or 1. p_c outside [0,1] is clamped into the
// open interval (pClamp, 1−pClamp).
func DME(pc float64) float64 {
	pc = clampUnit(pc)
	// p₀ = 1−p₀ = 1/2, so log(pc/p₀) = log(2·pc) and log((1−pc)/(1−p₀)) =
	// log(2·(1−pc)). Computed via the general form for clarity.
	term1 := pc * math.Log(pc/p0)
	term2 := (1 - pc) * math.Log((1-pc)/(1-p0))
	d := term1 + term2
	if d < 0 {
		// Guard against tiny negative values from floating-point error;
		// the KL is non-negative by Gibbs' inequality.
		d = 0
	}
	return d
}

// Weight returns the MGPO advantage weight w_ME(p_c) = exp(−λ·D_ME(p_c‖1/2)).
// λ must be ≥ 0. At λ = 0 the weight is exactly 1 for every p_c, so MGPO
// reduces to GRPO. The result lies in (0, 1], peaking at 1 when p_c = 1/2.
func Weight(lambda, pc float64) (float64, error) {
	if lambda < 0 {
		return 0, fmt.Errorf("mgpo: lambda must be >= 0, got %v", lambda)
	}
	if math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		return 0, fmt.Errorf("mgpo: lambda must be finite, got %v", lambda)
	}
	return weight(lambda, pc), nil
}

// weight is the unchecked core of Weight.
func weight(lambda, pc float64) float64 {
	if lambda == 0 {
		// Exact 1, independent of p_c: this is what makes MGPO ≡ GRPO at
		// λ=0 hold bit-for-bit.
		return 1
	}
	return math.Exp(-lambda * DME(pc))
}

// Accuracy returns the empirical accuracy p_c = (1/G) Σ I(rᵢ = 1) of a group of
// rewards, treating any reward > 0 as a success. It returns 0 for an empty
// group.
func Accuracy(rewards []float64) float64 {
	if len(rewards) == 0 {
		return 0
	}
	var n int
	for _, r := range rewards {
		if r > 0 {
			n++
		}
	}
	return float64(n) / float64(len(rewards))
}

func clampUnit(pc float64) float64 {
	if pc < pClamp {
		return pClamp
	}
	if pc > 1-pClamp {
		return 1 - pClamp
	}
	return pc
}
