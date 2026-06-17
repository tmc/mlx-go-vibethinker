package probe

import (
	"fmt"
	"math"
)

// PassK returns the unbiased Pass@K estimate for a single probe query: given
// n samples drawn and c of them correct, the probability that at least one of a
// random size-k subset is correct (Chen et al., the HumanEval estimator),
//
//	pass@k = 1 − C(n−c, k) / C(n, k).
//
// PassK uses the numerically-stable product form
//
//	pass@k = 1 − Π_{i=n−c+1}^{n} (1 − k/i),
//
// which avoids forming the large binomial coefficients directly. The closed
// form is defined only when k ≤ n; when n−c < k every size-k subset must hit a
// correct sample, so pass@k = 1. The two boundary cases follow: c = 0 ⇒ 0 and
// c = n ⇒ 1.
//
// It requires 0 ≤ c ≤ n, n ≥ 1, and 1 ≤ k ≤ n. Other arguments are an error.
func PassK(n, c, k int) (float64, error) {
	if n < 1 {
		return 0, fmt.Errorf("n must be >= 1, got %d", n)
	}
	if c < 0 || c > n {
		return 0, fmt.Errorf("c must be in [0, n]=[0,%d], got %d", n, c)
	}
	if k < 1 || k > n {
		return 0, fmt.Errorf("k must be in [1, n]=[1,%d], got %d", n, k)
	}
	return passK(n, c, k), nil
}

// passK is the unchecked core of PassK. It assumes n ≥ 1, 0 ≤ c ≤ n, and
// 1 ≤ k ≤ n.
func passK(n, c, k int) float64 {
	// If fewer than k samples are wrong, no size-k subset can avoid a correct
	// sample, so the estimate is exactly 1. This also covers c = n.
	if n-c < k {
		return 1
	}
	// c = 0 leaves the product empty over i = n+1..n; here n−c = n ≥ k, so the
	// branch above is not taken and the product runs i = n+1..n, which is
	// empty, giving pass@k = 1 − 1 = 0. Compute it directly for clarity.
	if c == 0 {
		return 0
	}
	// Stable product form: Π_{i=n−c+1}^{n} (1 − k/i) is the probability that a
	// random size-k subset draws only from the n−c wrong samples, i.e. the
	// failure probability C(n−c, k)/C(n, k).
	fail := 1.0
	for i := n - c + 1; i <= n; i++ {
		fail *= 1 - float64(k)/float64(i)
	}
	p := 1 - fail
	// Clamp to [0,1] to absorb floating-point error at the extremes.
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return p
}

// passKExact computes pass@k = 1 − C(n−c, k)/C(n, k) directly from the
// combinatorial closed form, in log space for numerical stability. It is the
// reference the product form is checked against; the stable product form is
// used in production. It assumes the same preconditions as passK.
func passKExact(n, c, k int) float64 {
	if n-c < k {
		return 1
	}
	// log C(n−c, k) − log C(n, k) = logChoose(n−c,k) − logChoose(n,k).
	logFail := logChoose(n-c, k) - logChoose(n, k)
	p := 1 - math.Exp(logFail)
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return p
}

// logChoose returns log C(a, b) via lgamma. It assumes 0 ≤ b ≤ a.
func logChoose(a, b int) float64 {
	lg := func(x float64) float64 { v, _ := math.Lgamma(x); return v }
	return lg(float64(a)+1) - lg(float64(b)+1) - lg(float64(a-b)+1)
}
