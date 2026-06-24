//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"time"
)

// SweepResult is one config's measured outcome: the held-out greedy Avg@1 at
// step 0 and at the final step (and the delta), the mechanism stats from the
// training run alongside as corroborating "why", and the run cost. AccStep0 and
// AccFinal are the DIRECTIONAL correctness signal (greedy Avg@1 over the fixed
// held-out probe, NOT benchmark accuracy); DeltaAcc = AccFinal − AccStep0 is what
// the sweep ranks configs by.
type SweepResult struct {
	Config     string  `json:"config"`
	Source     string  `json:"source"`
	Seed       uint64  `json:"seed"`
	AccStep0   float64 `json:"acc_step0"` // held-out greedy Avg@1 at the base policy
	AccFinal   float64 `json:"acc_final"` // held-out greedy Avg@1 after the training steps
	DeltaAcc   float64 `json:"delta_acc"` // AccFinal − AccStep0 (the ranking signal)
	HeldoutN   int     `json:"heldout_n"` // |held-out probe set|
	StepsRun   int     `json:"steps_run"` // real optimizer steps actually taken
	Mechanism  Metrics `json:"mechanism"` // mechanism + stability stats (corroborating)
	WallMillis float64 `json:"wall_millis"`
}

// RunSweptConfig runs ONE swept config end to end inside the current process:
// held-out greedy Avg@1 at step 0, the config's real GRPO training loop, then
// held-out Avg@1 at the final policy, returning the delta plus mechanism stats.
//
// IMPORTANT — this single-process path DOES NOT FIT under the Metal array ceiling
// on Qwen2.5-Math-1.5B: the non-reclaimable ~13GB training graph plus the O(n²)
// final held-out decode jointly exceed 499000 live buffers and crash in the final
// pass, even at the smoke floor (measured — see TestSweepBaselineHeldoutDelta).
// It is retained as the WALL DEMONSTRATOR and the building-block kernel; the real
// instrument is the TWO-PROCESS split (RunSweepPhase1 + RunSweepPhase2), which
// runs the final decode in a fresh process on a clean Metal budget. Prefer the
// split for any actual measurement.
//
// The held-out passes use the live model weights (Forward reads m.LM, which the
// trainer commits durable copies into after training — see trainer.commitDurable,
// which must run before the optimizer frees its params), so AccFinal genuinely
// reflects the post-training policy, not the base.
func RunSweptConfig(ctx context.Context, dir string, idx int, cfg Config, src Source) (SweepResult, error) {
	cfgs := SweepConfigs()
	if idx < 0 || idx >= len(cfgs) {
		return SweepResult{}, fmt.Errorf("realmodel: swept config index %d out of range [0,%d)", idx, len(cfgs))
	}
	sc := cfgs[idx]
	start := time.Now()

	m, err := Load(ctx, dir)
	if err != nil {
		return SweepResult{}, fmt.Errorf("realmodel: load for %q: %w", sc.Name, err)
	}
	defer func() {
		m.Close()
		reclaim()
	}()

	// --- Held-out @ step 0 (base policy), on the fixed probe. ---
	set := HeldoutSet()
	step0, err := scoreHeldout(ctx, m, set, heldoutMaxTokens)
	if err != nil {
		return SweepResult{}, fmt.Errorf("realmodel: held-out step0 for %q: %w", sc.Name, err)
	}
	reclaim()

	// --- Real GRPO training for this config. ---
	groups, err := buildGroups(ctx, m, cfg, src)
	if err != nil {
		return SweepResult{}, fmt.Errorf("realmodel: build %s groups for %q: %w", src, sc.Name, err)
	}
	groupsFreed := false
	defer func() {
		if !groupsFreed {
			freeGroups(groups)
		}
	}()
	mech, err := runMethod(ctx, m, sc.Method, cfg, groups)
	if err != nil {
		return SweepResult{}, fmt.Errorf("realmodel: train %q (%s): %w", sc.Name, src, err)
	}
	mech.Source = src.String()

	// Release the training phase's resident arrays — the frozen old/ref snapshots
	// — BEFORE the final held-out decode. The held-out pass is an O(n²) no-cache
	// re-forward that climbs toward the device array ceiling on its own; it must
	// start from the same clean budget the step-0 pass had, not on top of the
	// training graph's residue. (The optimizer's params were already detached into
	// model-owned copies and freed by runMethod's deferred commitDurable+free.)
	freeGroups(groups)
	groupsFreed = true
	reclaim()

	// --- Held-out @ final policy, same fixed probe. ---
	final, err := scoreHeldout(ctx, m, set, heldoutMaxTokens)
	if err != nil {
		return SweepResult{}, fmt.Errorf("realmodel: held-out final for %q: %w", sc.Name, err)
	}

	return SweepResult{
		Config:     sc.Name,
		Source:     src.String(),
		Seed:       cfg.Seed,
		AccStep0:   step0.Acc,
		AccFinal:   final.Acc,
		DeltaAcc:   final.Acc - step0.Acc,
		HeldoutN:   step0.N,
		StepsRun:   mech.Steps,
		Mechanism:  mech,
		WallMillis: float64(time.Since(start).Milliseconds()),
	}, nil
}

// heldoutMaxTokens bounds the held-out greedy decode in the sweep. It matches the
// probe test's cap (heldoutProbeMaxTokens) so the step-0 number the sweep reports
// is the same instrument the CKPT-A gate validated. The \boxed{} early-stop
// usually cuts well under this; the cap only bounds a runaway CoT.
const heldoutMaxTokens = 80
