package clr

import "context"

// FakeVerifier is a deterministic in-repo stand-in for the gated, model-prompted
// CLR Verifier. It returns a fixed set of trajectories for every prompt, padding
// or truncating each trajectory's claims to m and the trajectory count to k by
// repetition, so it can satisfy the (k, m) contract for any requested shape. It
// is used by the property tests and as a runnable example; a real run plugs in a
// model-backed Verifier instead.
type FakeVerifier struct {
	// Trajs is the template trajectory set. Score requests k trajectories with
	// m claims; FakeVerifier cycles through Trajs to reach k and pads/truncates
	// each claim slice (with 1s) to length m.
	Trajs []Trajectory
}

// Trajectories implements Verifier. It returns exactly k trajectories, each with
// exactly m claim verdicts, derived deterministically from the template.
func (f FakeVerifier) Trajectories(_ context.Context, _ string, k, m int) ([]Trajectory, error) {
	out := make([]Trajectory, k)
	n := len(f.Trajs)
	for i := 0; i < k; i++ {
		var src Trajectory
		if n > 0 {
			src = f.Trajs[i%n]
		}
		claims := make([]int, m)
		for j := 0; j < m; j++ {
			if j < len(src.Claims) {
				claims[j] = src.Claims[j]
			} else {
				claims[j] = 1 // pad with valid claims
			}
		}
		out[i] = Trajectory{Answer: src.Answer, Claims: claims}
	}
	return out, nil
}
