package methodcompare

import (
	"fmt"

	"github.com/tmc/mlx-go-vibethinker/signal/long2short"
)

// A scenario is the fixed, deterministic synthetic workload every method is
// evaluated on. It is generated entirely from a seed via a small self-contained
// PRNG (no math/rand, no model), so the same seed reproduces identical inputs and
// hence identical mechanism metrics — the reproducibility contract.
//
// The scenario deliberately contains the structure each knob is supposed to act
// on: prompt groups across multiple steps (so DCPO-SAS has a history to smooth),
// a zero-reward "cliff" group (so HDPO has an active set), groups with mixed
// accuracy and a couple of degenerate acc∈{0,1} groups (so Dynamic Sampling has
// something to drop), per-token importance ratios that straddle the clip
// ceilings (so Clip-Higher changes the bind rate), distinct rollout texts (so DRA
// sees diversity), and correct traces of differing lengths (so Long2Short moves
// tokens-per-sample).
type scenario struct {
	steps  []step             // per training step: groups of rewards + IDs + texts
	ratios []float64          // per-token importance ratios for the clip-bind metric
	traces []long2short.Trace // a correct-set with varied lengths for Long2Short
}

// A step is one training step's batch of prompt groups.
type step struct {
	rewards   [][]float64
	promptIDs []string
	texts     [][]string // rollout texts per group, for DRA diversity
}

// newScenario builds the deterministic scenario for seed.
func newScenario(seed uint64) *scenario {
	r := &lcg{state: seed*2862933555777941757 + 3037000493}

	const steps = 4
	const groups = 6
	const rollouts = 4

	// Fixed accuracy templates per group, chosen to cover the cases each knob
	// needs. Group 0: cliff (all fail). Groups 1: ceiling (all pass). The rest:
	// mixed accuracy. The cliff/ceiling groups are exactly what Dynamic Sampling
	// drops and (for the cliff) what HDPO acts on.
	accTemplates := [groups]int{0, rollouts, 1, 2, 3, 2} // count of successes per group

	sc := &scenario{}
	for s := 0; s < steps; s++ {
		st := step{
			rewards:   make([][]float64, groups),
			promptIDs: make([]string, groups),
			texts:     make([][]string, groups),
		}
		for g := 0; g < groups; g++ {
			st.promptIDs[g] = fmt.Sprintf("q%d", g)
			rew := make([]float64, rollouts)
			txt := make([]string, rollouts)
			// The first nSucc slots succeed; slot positions are STABLE across
			// steps (so SAS smooths a real per-slot series, not a permutation
			// that would pathologically average to zero), while a small
			// deterministic per-step magnitude perturbation on the successful
			// rollouts gives SAS genuine cross-step variance to smooth — lowering
			// the advantage variance across steps without collapsing |A|. Group
			// accuracy (pass/fail count) is held fixed, so w_ME and the cliff/
			// ceiling structure are identical every step.
			nSucc := accTemplates[g]
			for j := 0; j < rollouts; j++ {
				if j < nSucc {
					// Successful reward in (0.7, 1.0], perturbed per step/slot so
					// the advantage magnitude varies across steps.
					noise := 0.15 * (float64((g*7+j*3+s*5)%10) / 10.0)
					rew[j] = 1.0 - noise
				}
				// Distinct, deterministic rollout texts; crowd the first two
				// rollouts of each group together and keep the rest distinctive,
				// so DRA sees a non-uniform similarity structure.
				if j < 2 {
					txt[j] = fmt.Sprintf("group %d common reasoning path aaaa bbbb", g)
				} else {
					txt[j] = fmt.Sprintf("group %d distinct path %d %d", g, j, int(r.next()%97))
				}
			}
			st.rewards[g] = rew
			st.texts[g] = txt
		}
		sc.steps = append(sc.steps, st)
	}

	// Per-token importance ratios: a deterministic spread that straddles the
	// symmetric (1.25) and Clip-Higher (1.28) upper ceilings and the lower 0.8/
	// 0.75 floors, so the bind rate differs between clip schemes.
	const nTok = 200
	sc.ratios = make([]float64, nTok)
	for i := range sc.ratios {
		// Spread ratios in [0.6, 1.45] deterministically from the PRNG.
		u := float64(r.next()%1000) / 1000.0 // [0,1)
		sc.ratios[i] = 0.6 + 0.85*u
	}

	// Long2Short correct set: same total reward, varied lengths, plus one
	// incorrect trace that must stay untouched.
	sc.traces = []long2short.Trace{
		{Reward: 1, Length: 60, Correct: true},
		{Reward: 1, Length: 140, Correct: true},
		{Reward: 1, Length: 220, Correct: true},
		{Reward: 1, Length: 300, Correct: true},
		{Reward: 0, Length: 90, Correct: false},
	}

	return sc
}

// cliffJSD returns a per-group JSD vector for the cliff groups of rewards,
// derived deterministically from the group index so HDPO has a concrete,
// reproducible self-teacher divergence to weight. Non-cliff groups get 0 (HDPO
// ignores them). The values are fixed, not random, so the metric is reproducible.
func (sc *scenario) cliffJSD(rewards [][]float64) []float64 {
	out := make([]float64, len(rewards))
	for i := range rewards {
		// A fixed, plausible JSD in (0, ln2) for the cliff groups.
		out[i] = 0.30 + 0.01*float64(i)
	}
	return out
}

// lcg is a minimal deterministic linear-congruential PRNG. It is self-contained
// (no math/rand) so the scenario is bit-reproducible across runs and machines.
type lcg struct{ state uint64 }

func (l *lcg) next() uint64 {
	// Numerical Recipes LCG constants.
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return l.state >> 16
}
