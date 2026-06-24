//go:build modelir

package realmodel

import (
	"context"
	"testing"
)

// heldoutProbeMaxTokens bounds the greedy decode in the probe and the sweep. The
// base Qwen2.5-Math is VERBOSE: it emits a multi-step rationale before the
// \boxed{} answer, so too tight a budget truncates correct reasoning before the
// boxed result and undercounts accuracy. 80 lets the CoT complete on this
// short-horizon set while staying well under the Metal array ceiling: the
// no-cache greedy decode is O(n²) and its live Metal arrays are not reclaimable
// in-process (see generateGreedy), so the cap plus the \boxed{} early-stop keeps
// a full step-0+final held-out pass inside one config's subprocess.
const heldoutProbeMaxTokens = 80

// TestHeldoutStep0BaselineHasDynamicRange is the CKPT-A gate: it measures the
// base model's step-0 greedy Avg@1 on the FIXED held-out set and asserts the
// score sits meaningfully OFF BOTH FLOORS (not 0%, not 100%). If the held-out
// signal is pinned at a floor, Δacc has no dynamic range and the whole sweep
// cannot separate methods — that must be caught HERE, not at the final report.
//
// It logs the per-prompt scores so the held-out difficulty can be tuned: the set
// is graded (easy / medium / hard blocks) to land the base strictly between the
// floors under greedy decode.
func TestHeldoutStep0BaselineHasDynamicRange(t *testing.T) {
	m := requireModel(t)
	ctx := context.Background()
	set := HeldoutSet()

	res, err := scoreHeldout(ctx, m, set, heldoutProbeMaxTokens)
	if err != nil {
		t.Fatalf("scoreHeldout: %v", err)
	}

	var nRight, nWrong int
	for i, s := range set {
		mark := "WRONG"
		if res.Scores[i] == 1 {
			mark = "right"
			nRight++
		} else {
			nWrong++
		}
		t.Logf("  [%2d] %-5s gold=%-6s | %s", i, mark, s.Gold, s.Question)
	}
	t.Logf("HELD-OUT step-0 baseline greedy Avg@1 = %.3f (%d/%d right) over N=%d",
		res.Acc, nRight, len(set), res.N)

	// The dynamic-range gate: off both floors so Δacc can move up OR down.
	if res.Acc <= 0 {
		t.Fatalf("step-0 baseline Avg@1 = %.3f is at the FLOOR (0%%): no dynamic range for Δacc; "+
			"the held-out set is too hard for the base — rethink the signal (CKPT-A blocker)", res.Acc)
	}
	if res.Acc >= 1 {
		t.Fatalf("step-0 baseline Avg@1 = %.3f is at the CEILING (100%%): no dynamic range for Δacc; "+
			"the held-out set is too easy — add harder prompts (CKPT-A blocker)", res.Acc)
	}
	// Stronger: we want it meaningfully off the floors (some headroom both ways),
	// not just barely. A warn (not fatal) if it is in the extreme decile, so the
	// CKPT-A reviewer sees the margin.
	if res.Acc < 0.15 || res.Acc > 0.85 {
		t.Logf("WARNING: step-0 Avg@1 = %.3f is near a floor (<0.15 or >0.85) — limited dynamic range; "+
			"consider regrading the held-out set for more headroom", res.Acc)
	}
}
