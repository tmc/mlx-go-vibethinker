package mgpo

import (
	"testing"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// Phase-B property (DESIGN_RL_UPGRADE.md §2 Tier 2, DAPO Dynamic Sampling):
// off-path bit-identical. A batch whose groups are all learnable (0<acc<1)
// passes through DynamicSample unchanged — same groups, same order, same IDs —
// and the advantage of any kept group is identical to the un-filtered advantage.
// The control half (a batch containing degenerate groups) must drop exactly the
// degenerate ones, proving the knob is live.
func TestDynamicSampleOffPathBitIdentical(t *testing.T) {
	// All learnable: 0 < acc < 1 in every group.
	rewards := [][]float64{
		{1, 0, 1, 1}, // acc 0.75
		{0, 0, 1, 0}, // acc 0.25
		{1, 1, 0, 0}, // acc 0.5
	}
	ids := []string{"a", "b", "c"}

	keptR, keptIDs := DynamicSample(rewards, ids)
	if len(keptR) != len(rewards) {
		t.Fatalf("learnable batch shrank: kept %d of %d groups", len(keptR), len(rewards))
	}
	for i := range rewards {
		if keptIDs[i] != ids[i] {
			t.Fatalf("group %d id reordered: %q != %q", i, keptIDs[i], ids[i])
		}
		for j := range rewards[i] {
			if keptR[i][j] != rewards[i][j] {
				t.Fatalf("group %d slot %d reward changed: %v != %v", i, j, keptR[i][j], rewards[i][j])
			}
		}
	}

	// The kept batch's advantage is bit-identical to the un-filtered advantage.
	wantAdv, err := ScaledAdvantagesOpt(rewards, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt unfiltered: %v", err)
	}
	gotAdv, err := ScaledAdvantagesOpt(keptR, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt filtered: %v", err)
	}
	assertEqualGroups(t, gotAdv, wantAdv, "dynamic-sample off-path advantage")

	// Control: with degenerate groups present, the knob removes exactly them.
	mixed := [][]float64{
		{1, 1, 1, 1}, // acc 1 — degenerate, dropped
		{1, 0, 1, 0}, // acc 0.5 — kept
		{0, 0, 0, 0}, // acc 0 — degenerate, dropped
		{0, 1, 0, 0}, // acc 0.25 — kept
	}
	mixedIDs := []string{"all-pass", "mid", "all-fail", "low"}
	keptMixed, keptMixedIDs := DynamicSample(mixed, mixedIDs)
	if len(keptMixed) != 2 {
		t.Fatalf("dynamic sample kept %d groups, want 2 learnable", len(keptMixed))
	}
	if keptMixedIDs[0] != "mid" || keptMixedIDs[1] != "low" {
		t.Fatalf("dynamic sample kept the wrong groups / order: %v", keptMixedIDs)
	}
}

// Phase-B property: Dynamic Sampling unifies with the std=0 zero-advantage
// guard. Every group it drops (acc ∈ {0,1}) is exactly a group whose
// std-normalized advantage is all zeros, and every group it keeps has a nonzero
// advantage — so dropping costs no learning signal that the loss would have
// used.
func TestDynamicSampleUnifiesWithStdZeroGuard(t *testing.T) {
	groups := [][]float64{
		{1, 1, 1, 1}, // degenerate
		{1, 0, 1, 1}, // learnable
		{0, 0, 0, 0}, // degenerate
		{1, 0, 0, 0}, // learnable
	}
	for i, g := range groups {
		adv := rl.GroupAdvantage([][]float64{g})[0]
		allZero := true
		for _, a := range adv {
			if a != 0 {
				allZero = false
				break
			}
		}
		learnable := Learnable(g)
		// Learnable iff the std-normalized advantage is NOT all-zero.
		if learnable == allZero {
			t.Fatalf("group %d %v: Learnable=%v but std-advantage allZero=%v — guard not unified",
				i, g, learnable, allZero)
		}
	}
}

// Phase-B cross-cutting invariant: filtering preserves the MGPO no-op rule on
// the kept groups — the per-group advantage and w_ME scaling of a retained group
// is exactly what it would be without filtering, because the filter only removes
// whole groups and never touches a kept group's rewards.
func TestDynamicSampleKeepsNoOpRule(t *testing.T) {
	const lambda = 2.0
	full := [][]float64{
		{1, 1, 1, 1}, // dropped
		{1, 0, 1, 1}, // kept (acc 0.75)
	}
	kept, _ := DynamicSample(full, nil)
	if len(kept) != 1 {
		t.Fatalf("kept %d groups, want 1", len(kept))
	}
	// The kept group's scaled advantage equals scaling that group alone.
	gotBatch, err := ScaledAdvantagesOpt(kept, lambda, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt kept: %v", err)
	}
	gotAlone, err := ScaledAdvantagesOpt([][]float64{full[1]}, lambda, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt alone: %v", err)
	}
	assertEqualGroups(t, gotBatch, gotAlone, "dynamic-sample kept group no-op")
}
