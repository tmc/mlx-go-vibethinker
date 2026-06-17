package clr

import (
	"context"
	"fmt"
	"math"
)

// Default CLR parameters from the paper (DESIGN §4.7): K=32 trajectories,
// M=5 claims per trajectory, R=8 repetitions of the whole flow.
const (
	DefaultK = 32
	DefaultM = 5
	DefaultR = 8
)

// A Trajectory is one model rollout under CLR: its final answer plus the
// per-claim self-verification verdicts v_{k,m} ∈ {0,1}. Claims holds exactly M
// verdicts, each 0 or 1.
type Trajectory struct {
	Answer string
	Claims []int
}

// A Verifier produces the CLR trajectories for one prompt: it generates K
// rollouts, extracts M decision-relevant claims and a final answer from each,
// and self-verifies the claims. These are model-prompted steps, so the Verifier
// is a gated seam; FakeVerifier provides a deterministic in-package stand-in for
// tests. It must return exactly k trajectories, each with m claim verdicts.
type Verifier interface {
	Trajectories(ctx context.Context, prompt string, k, m int) ([]Trajectory, error)
}

// An Equivalence reports whether two final answers denote the same result. It
// must be reflexive and symmetric; CLR uses it to cluster trajectories. A nil
// Equivalence defaults to exact string equality.
type Equivalence func(a, b string) bool

// reliability is the numeric core for one trajectory: r_k = ((1/M)Σ v_m)^M,
// with v_m ∈ {0,1}. It returns an error if any verdict is not 0 or 1 or claims
// is empty. r_k ∈ [0,1], equals 1 iff every claim is valid, and 0 if any whole
// claim is missing (mean < 1 raised to M shrinks fast).
func reliability(claims []int) (float64, error) {
	m := len(claims)
	if m == 0 {
		return 0, fmt.Errorf("clr: trajectory has no claims")
	}
	var valid int
	for i, v := range claims {
		switch v {
		case 0:
		case 1:
			valid++
		default:
			return 0, fmt.Errorf("clr: claim %d verdict %d not in {0,1}", i, v)
		}
	}
	mean := float64(valid) / float64(m)
	return math.Pow(mean, float64(m)), nil
}

// selectAnswer is the numeric core for one CLR flow: cluster trajectories by
// answer equivalence and return the answer of the cluster maximizing Σ_{k∈G} r_k
// (DESIGN §4.7). It returns the chosen answer and its summed reliability. Ties
// are broken by first appearance, so the result is deterministic. trajs must be
// non-empty; eq defaults to exact equality when nil.
func selectAnswer(trajs []Trajectory, eq Equivalence) (string, float64, error) {
	if len(trajs) == 0 {
		return "", 0, fmt.Errorf("clr: no trajectories")
	}
	if eq == nil {
		eq = func(a, b string) bool { return a == b }
	}
	// Clusters in first-appearance order; sums parallel them.
	reps := make([]string, 0, len(trajs))
	sums := make([]float64, 0, len(trajs))
	for _, tr := range trajs {
		r, err := reliability(tr.Claims)
		if err != nil {
			return "", 0, err
		}
		placed := false
		for i, rep := range reps {
			if eq(rep, tr.Answer) {
				sums[i] += r
				placed = true
				break
			}
		}
		if !placed {
			reps = append(reps, tr.Answer)
			sums = append(sums, r)
		}
	}
	best := 0
	for i := 1; i < len(sums); i++ {
		if sums[i] > sums[best] {
			best = i
		}
	}
	return reps[best], sums[best], nil
}

// Reliability returns the CLR reliability r_k of a single trajectory,
// r_k = ((1/M)Σ v_m)^M (DESIGN §4.7). It is exported so callers and tests can
// pin the invariant directly.
func Reliability(t Trajectory) (float64, error) {
	return reliability(t.Claims)
}

// Result reports a CLR evaluation over R flows: the selected answer of each
// flow, whether each matched the gold answer, and the mean Pass@1.
type Result struct {
	Answers  []string
	Correct  []bool
	MeanPass float64
}

// Score runs the CLR flow R times for one prompt and reports the mean Pass@1 of
// the selected answer against gold (DESIGN §4.7). Each flow asks verifier for k
// trajectories of m claims, computes per-trajectory reliability, clusters
// answers by eq, and selects the cluster maximizing summed reliability. k, m,
// and r must be ≥ 1; verifier must be non-nil; eq defaults to exact equality.
func Score(ctx context.Context, verifier Verifier, prompt, gold string, k, m, r int, eq Equivalence, correct Equivalence) (Result, error) {
	if verifier == nil {
		return Result{}, fmt.Errorf("clr: nil verifier")
	}
	if k < 1 || m < 1 || r < 1 {
		return Result{}, fmt.Errorf("clr: k, m, r must be >= 1, got k=%d m=%d r=%d", k, m, r)
	}
	if correct == nil {
		correct = func(a, b string) bool { return a == b }
	}
	res := Result{
		Answers: make([]string, r),
		Correct: make([]bool, r),
	}
	var hits float64
	for i := 0; i < r; i++ {
		trajs, err := verifier.Trajectories(ctx, prompt, k, m)
		if err != nil {
			return Result{}, fmt.Errorf("clr: flow %d trajectories: %w", i, err)
		}
		if len(trajs) != k {
			return Result{}, fmt.Errorf("clr: flow %d returned %d trajectories, want %d", i, len(trajs), k)
		}
		for j, tr := range trajs {
			if len(tr.Claims) != m {
				return Result{}, fmt.Errorf("clr: flow %d trajectory %d has %d claims, want %d", i, j, len(tr.Claims), m)
			}
		}
		ans, _, err := selectAnswer(trajs, eq)
		if err != nil {
			return Result{}, fmt.Errorf("clr: flow %d select: %w", i, err)
		}
		res.Answers[i] = ans
		ok := correct(ans, gold)
		res.Correct[i] = ok
		if ok {
			hits++
		}
	}
	res.MeanPass = hits / float64(r)
	return res, nil
}
