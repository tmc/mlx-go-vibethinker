package long2short

import (
	"fmt"
	"math"
)

// DefaultLambda is the paper's Long2Short reshaping coefficient (λ = 0.2).
const DefaultLambda = 0.2

// A Trace is one rollout in a prompt group: its reward and its length in
// tokens. Correct identifies whether the trace belongs to the correct set C
// (the brevity reshaping applies only to C).
type Trace struct {
	Reward  float64
	Length  int
	Correct bool
}

// Reshape returns the Long2Short-reshaped rewards for one prompt group, in the
// same order as traces. Incorrect traces keep their reward. Among the correct
// set C it adds the zero-sum brevity shift
//
//	λ·(sᵢ − s̄)/max_{j∈C}|sⱼ − s̄|,   sᵢ = 1/Lᵢ,
//
// where s̄ is the mean brevity over C and the max is over C only. If C is empty,
// has one element, or all its lengths are equal (zero denominator), the rewards
// are returned unchanged. lambda must be ≥ 0; a correct trace with non-positive
// length is an error.
func Reshape(traces []Trace, lambda float64) ([]float64, error) {
	if lambda < 0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		return nil, fmt.Errorf("long2short: lambda must be finite and >= 0, got %v", lambda)
	}
	// Gather brevity scores for the correct set and validate lengths.
	idx := make([]int, 0, len(traces))
	brev := make([]float64, 0, len(traces))
	for i, t := range traces {
		if !t.Correct {
			continue
		}
		if t.Length <= 0 {
			return nil, fmt.Errorf("long2short: correct trace %d has non-positive length %d", i, t.Length)
		}
		idx = append(idx, i)
		brev = append(brev, 1.0/float64(t.Length))
	}
	out := make([]float64, len(traces))
	for i, t := range traces {
		out[i] = t.Reward
	}
	shifts, ok := reshapeShifts(brev, lambda)
	if !ok {
		return out, nil // no-op: empty/singleton C or equal lengths
	}
	for k, i := range idx {
		out[i] += shifts[k]
	}
	return out, nil
}

// reshapeShifts computes the zero-sum brevity shift for the correct-set brevity
// scores. It returns (shifts, true) when a shift applies, or (nil, false) when
// the set is too small or all brevity scores are equal (zero denominator). The
// returned shifts sum to zero up to floating-point error.
func reshapeShifts(brev []float64, lambda float64) ([]float64, bool) {
	n := len(brev)
	if n < 2 {
		return nil, false
	}
	var sum float64
	for _, s := range brev {
		sum += s
	}
	mean := sum / float64(n)

	var maxDev float64
	dev := make([]float64, n)
	for i, s := range brev {
		dev[i] = s - mean
		if a := math.Abs(dev[i]); a > maxDev {
			maxDev = a
		}
	}
	if maxDev == 0 {
		return nil, false // all lengths equal
	}
	shifts := make([]float64, n)
	for i := range brev {
		shifts[i] = lambda * dev[i] / maxDev
	}
	return shifts, true
}
