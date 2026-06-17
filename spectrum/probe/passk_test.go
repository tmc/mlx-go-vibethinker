package probe

import (
	"math"
	"math/big"
	"testing"
)

// chooseBig returns C(a, b) as an exact big.Int (0 for b < 0 or b > a).
func chooseBig(a, b int) *big.Int {
	if b < 0 || b > a {
		return big.NewInt(0)
	}
	return new(big.Int).Binomial(int64(a), int64(b))
}

// exactPassK computes pass@k = 1 − C(n−c,k)/C(n,k) as an exact rational, the
// ground truth the stable product form is checked against. It assumes the same
// preconditions as passK.
func exactPassK(n, c, k int) *big.Rat {
	if n-c < k { // every size-k subset hits a correct sample
		return big.NewRat(1, 1)
	}
	num := chooseBig(n-c, k)
	den := chooseBig(n, k)
	fail := new(big.Rat).SetFrac(num, den)
	return new(big.Rat).Sub(big.NewRat(1, 1), fail)
}

// Invariant (DESIGN §5 item 5): the stable product form matches the unbiased
// combinatorial closed form 1 − C(n−c,k)/C(n,k) for several (n,c,k) with n > k.
func TestPassKMatchesCombinatorialClosedForm(t *testing.T) {
	cases := []struct{ n, c, k int }{
		{10, 0, 1}, {10, 1, 1}, {10, 5, 1}, {10, 10, 1},
		{10, 3, 2}, {10, 5, 3}, {10, 7, 4}, {10, 2, 5},
		{20, 1, 3}, {20, 4, 5}, {20, 13, 7}, {20, 19, 10},
		{64, 1, 32}, {64, 7, 16}, {64, 33, 8}, {100, 50, 10},
		{8, 3, 8}, // k == n
	}
	for _, tc := range cases {
		got, err := PassK(tc.n, tc.c, tc.k)
		if err != nil {
			t.Fatalf("PassK(%d,%d,%d): %v", tc.n, tc.c, tc.k, err)
		}
		want, _ := exactPassK(tc.n, tc.c, tc.k).Float64()
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("PassK(%d,%d,%d) = %.15g, want %.15g (Δ=%g)",
				tc.n, tc.c, tc.k, got, want, math.Abs(got-want))
		}
		// The lgamma reference path must agree too.
		if ex := passKExact(tc.n, tc.c, tc.k); math.Abs(ex-want) > 1e-9 {
			t.Errorf("passKExact(%d,%d,%d) = %.15g, want %.15g", tc.n, tc.c, tc.k, ex, want)
		}
	}
}

// Invariant (DESIGN §5 item 5): c = 0 ⇒ 0 and c = n ⇒ 1.
func TestPassKBoundaries(t *testing.T) {
	for _, n := range []int{1, 2, 5, 10, 64} {
		for _, k := range []int{1, n / 2, n} {
			if k < 1 {
				continue
			}
			zero, err := PassK(n, 0, k)
			if err != nil {
				t.Fatalf("PassK(%d,0,%d): %v", n, k, err)
			}
			if zero != 0 {
				t.Errorf("c=0: PassK(%d,0,%d) = %v, want 0", n, k, zero)
			}
			one, err := PassK(n, n, k)
			if err != nil {
				t.Fatalf("PassK(%d,%d,%d): %v", n, n, k, err)
			}
			if one != 1 {
				t.Errorf("c=n: PassK(%d,%d,%d) = %v, want 1", n, n, k, one)
			}
		}
	}
}

// When n−c < k the estimate is exactly 1: too few wrong samples to fill a
// size-k subset.
func TestPassKShortfallIsOne(t *testing.T) {
	// n=5, c=3 -> n-c=2 wrong; k=3 > 2, so every subset of 3 hits a correct.
	got, err := PassK(5, 3, 3)
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("PassK(5,3,3) = %v, want 1", got)
	}
}

// pass@k is monotone non-decreasing in c (more correct samples never lowers the
// chance a size-k subset hits one).
func TestPassKMonotoneInC(t *testing.T) {
	const n, k = 30, 4
	prev := -1.0
	for c := 0; c <= n; c++ {
		got, err := PassK(n, c, k)
		if err != nil {
			t.Fatal(err)
		}
		if got < prev-1e-12 {
			t.Fatalf("PassK(%d,%d,%d)=%v < prev %v: not monotone in c", n, c, k, got, prev)
		}
		prev = got
	}
}

func TestPassKValidation(t *testing.T) {
	bad := []struct {
		name    string
		n, c, k int
	}{
		{"n<1", 0, 0, 1},
		{"c<0", 5, -1, 1},
		{"c>n", 5, 6, 1},
		{"k<1", 5, 2, 0},
		{"k>n", 5, 2, 6},
	}
	for _, tc := range bad {
		if _, err := PassK(tc.n, tc.c, tc.k); err == nil {
			t.Errorf("%s: PassK(%d,%d,%d) want error", tc.name, tc.n, tc.c, tc.k)
		}
	}
}
