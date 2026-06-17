package mgpo

// DAPO Dynamic Sampling (arXiv 2503.14476) filters a batch of prompt groups
// before advantage construction, keeping only groups whose empirical accuracy is
// strictly between 0 and 1. A group that all rollouts pass (acc = 1) or all fail
// (acc = 0) has identical rewards, hence zero group variance and a zero
// group-relative advantage — it contributes no learning signal but still costs a
// forward/backward pass. Dropping it at the data layer concentrates the batch on
// groups that actually move the policy.
//
// This is the same degeneracy the std-normalized advantage already guards
// (GroupAdvantage returns zeros when std = 0, which happens exactly when the
// rewards are identical, i.e. acc ∈ {0,1}). Dynamic Sampling unifies with that
// guard: rather than feeding a zero-advantage group through the loss, it removes
// the group from the batch entirely. The filter lives upstream of the loss, so
// it changes which groups are optimized, never how a kept group's loss is
// computed — the per-group computation is bit-identical to today.

// Learnable reports whether a reward group carries a learning signal under
// group-relative advantage: its empirical accuracy is strictly inside (0, 1), so
// the rewards are not all-equal and the group variance is nonzero. An empty
// group is not learnable.
func Learnable(group []float64) bool {
	if len(group) == 0 {
		return false
	}
	acc := Accuracy(group)
	return acc > 0 && acc < 1
}

// DynamicSampleIndices returns the indices of the reward groups that survive
// DAPO Dynamic Sampling: those that are [Learnable]. The indices are in
// ascending order, so subsetting any parallel per-group slice (prompt IDs,
// rollout tensors) by these indices preserves alignment.
//
// With every group learnable the result is 0..len(rewards)-1 unchanged, so
// applying the filter to a batch that has no degenerate groups is a no-op and
// the downstream advantage/loss is bit-identical to not filtering at all.
func DynamicSampleIndices(rewards [][]float64) []int {
	keep := make([]int, 0, len(rewards))
	for i, g := range rewards {
		if Learnable(g) {
			keep = append(keep, i)
		}
	}
	return keep
}

// DynamicSample applies DAPO Dynamic Sampling to a batch, returning the kept
// reward groups and the kept prompt IDs in their original order. promptIDs may
// be nil (then the returned IDs are nil); otherwise it must have one entry per
// reward group. It is a thin convenience over [DynamicSampleIndices].
//
// When no group is degenerate the returned batch equals the input, so the
// off-path (a batch with only learnable groups, or the caller simply not
// invoking this) is identical to today. Filtering only ever removes
// zero-gradient groups, so it never changes a retained group's advantage or
// loss.
func DynamicSample(rewards [][]float64, promptIDs []string) (keptRewards [][]float64, keptIDs []string) {
	idx := DynamicSampleIndices(rewards)
	keptRewards = make([][]float64, len(idx))
	if promptIDs != nil {
		keptIDs = make([]string, len(idx))
	}
	for n, i := range idx {
		keptRewards[n] = rewards[i]
		if promptIDs != nil {
			keptIDs[n] = promptIDs[i]
		}
	}
	return keptRewards, keptIDs
}
