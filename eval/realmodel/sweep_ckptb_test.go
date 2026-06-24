//go:build modelir

package realmodel

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

// skipIfNoModel skips the test when the model weights are absent, WITHOUT loading
// the multi-GB model (unlike requireModel). The CKPT-B test must let
// RunSweptConfig do the single load itself — keeping two models resident OOMs the
// wired working set (see load.go).
func skipIfNoModel(t *testing.T) {
	t.Helper()
	dir := DefaultModelDir()
	if dir == "" {
		t.Skip("realmodel: no home directory to locate the model")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("realmodel: model not present at %q (set %s); skipping", dir, modelDirEnv)
	}
}

// envInt reads an integer knob from the environment with a default, for the
// CKPT-B ceiling search: the test runs the baseline at escalating compute knobs
// (prompts/k/max-tokens/steps) WITHOUT recompiling, so the largest green config
// can be found by re-running with different env values until one OOMs / trips the
// Metal array ceiling. The default is a small, known-green smoke.
func envInt(t *testing.T, key string, def int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("env %s=%q: %v", key, v, err)
	}
	return n
}

func envU64(t *testing.T, key string, def uint64) uint64 {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		t.Fatalf("env %s=%q: %v", key, v, err)
	}
	return n
}

// TestSweepBaselineHeldoutDelta DOCUMENTS THE SINGLE-PROCESS WALL: it runs the C1
// baseline config end to end IN ONE PROCESS — held-out greedy Avg@1 at step 0, a
// real GRPO training loop, held-out Avg@1 at the final step — via RunSweptConfig,
// the single-process kernel. On this Qwen2.5-Math-1.5B it CRASHES in the final
// held-out pass (Metal array ceiling 499000): the non-reclaimable ~13GB training
// graph plus the O(n²) final decode jointly bust the ceiling, even at the smoke
// floor (prompts=6/k=4/maxTok=32/steps=8). That measured wall is why the real
// instrument is the TWO-PROCESS split (RunSweepPhase1/RunSweepPhase2 + the -sweep
// parent), where each phase runs on a clean Metal budget.
//
// It is SKIPPED by default (it is expected to crash, and a crashing run reds the
// suite). Set VT_RUN_SINGLEPROC_WALL=1 to run it — the env knobs VT_PROMPTS /
// VT_K / VT_MAXTOK / VT_STEPS / VT_SEED / VT_SOURCE let you re-probe the wall at
// other sizes (it crashes at all of them; that is the point).
func TestSweepBaselineHeldoutDelta(t *testing.T) {
	if os.Getenv("VT_RUN_SINGLEPROC_WALL") == "" {
		t.Skip("single-process wall demonstrator — crashes by design (Metal ceiling); " +
			"set VT_RUN_SINGLEPROC_WALL=1 to run. The real instrument is the two-process -sweep split.")
	}
	skipIfNoModel(t) // skip-without-loading; RunSweptConfig does the single load
	ctx := context.Background()

	cfg := Config{
		Prompts:     envInt(t, "VT_PROMPTS", 6),
		K:           envInt(t, "VT_K", 4),
		MaxTokens:   envInt(t, "VT_MAXTOK", 32),
		Temperature: 0.8,
		Steps:       envInt(t, "VT_STEPS", 8),
		LR:          1e-6,
		Seed:        envU64(t, "VT_SEED", 1),
	}
	srcStr := os.Getenv("VT_SOURCE")
	if srcStr == "" {
		srcStr = "seeded"
	}
	src, err := ParseSource(srcStr)
	if err != nil {
		t.Fatalf("VT_SOURCE: %v", err)
	}

	t.Logf("CKPT-B baseline run: prompts=%d k=%d max-tokens=%d steps=%d seed=%d source=%s",
		cfg.Prompts, cfg.K, cfg.MaxTokens, cfg.Steps, cfg.Seed, src)

	res, err := RunSweptConfig(ctx, DefaultModelDir(), 0 /* C1 baseline */, cfg, src)
	if err != nil {
		t.Fatalf("RunSweptConfig(C1 baseline): %v", err)
	}

	t.Logf("HELD-OUT (N=%d) Avg@1: step0=%.3f final=%.3f  Δacc=%+.3f  (steps=%d, %.1fs)",
		res.HeldoutN, res.AccStep0, res.AccFinal, res.DeltaAcc, res.StepsRun, res.WallMillis/1000)
	t.Logf("MECHANISM: ratioMean=%.4f ratioVar=%.4f maxAbsDev=%.4f finalLoss=%v lossFinite=%v",
		res.Mechanism.RatioMean, res.Mechanism.RatioVar, res.Mechanism.RatioMaxAbsDev,
		floatPtr(res.Mechanism.FinalLoss), res.Mechanism.LossFinite)

	if !res.Mechanism.LossFinite {
		t.Fatalf("baseline loss not finite — instability, not a usable measurement")
	}
	if res.StepsRun == 0 {
		t.Logf("WARNING: 0 optimizer steps taken (all groups dropped?) — Δacc cannot move; "+
			"check source/config (source=%s)", src)
	}
}

func floatPtr(p *float64) string {
	if p == nil {
		return "nil"
	}
	return strconv.FormatFloat(*p, 'f', 4, 64)
}

// maxAbsDiff returns max|a−b| as a float64, casting to float32 first (q/v weights
// are bf16). It is the round-trip fidelity metric: 0 means bit-exact.
func maxAbsDiff(t *testing.T, a, b *mlx.Array) float64 {
	t.Helper()
	af, err := mlx.AsType[float32](a)
	if err != nil {
		t.Fatalf("astype a: %v", err)
	}
	defer af.Free()
	bf, err := mlx.AsType[float32](b)
	if err != nil {
		t.Fatalf("astype b: %v", err)
	}
	defer bf.Free()
	d := mlx.Abs(mlx.Subtract(af, bf))
	defer d.Free()
	m := mlx.Max(d, false)
	defer m.Free()
	if err := mlx.Eval(m); err != nil {
		t.Fatalf("eval maxabsdiff: %v", err)
	}
	return float64(mlx.ArrayItemFloat32(m))
}

// TestCKPTBCheckpointRoundTripsExact is REQ#2: the q/v checkpoint must round-trip
// bit-for-bit. It captures the base q/v, saves a checkpoint, zeroes the model's
// q/v, reloads the checkpoint, and asserts the reloaded weights equal the captured
// base EXACTLY (max|Δ|==0) on every tensor — verifying the symmetric slot Set/Get
// round-trip needs no HF-suffix/transpose translation (it is NOT assumed). It uses
// ONE model (two resident multi-GB models OOM the wired set).
func TestCKPTBCheckpointRoundTripsExact(t *testing.T) {
	m := requireModel(t)
	defer func() { m.Close(); reclaim() }()

	// Capture the base q/v values (deep copies, evaluated) keyed by slot path.
	slots, base, err := snapshotParams(m)
	if err != nil {
		t.Fatalf("snapshotParams: %v", err)
	}
	defer func() {
		for _, a := range base {
			a.Free()
		}
	}()
	baseByPath := make(map[string]*mlx.Array, len(slots))
	for i, s := range slots {
		baseByPath[s.path] = base[i]
	}

	ckpt := filepath.Join(t.TempDir(), "qv.safetensors")
	n, err := saveTrainedQV(m, ckpt)
	if err != nil {
		t.Fatalf("saveTrainedQV: %v", err)
	}
	if n != len(slots) {
		t.Fatalf("checkpoint wrote %d tensors, want %d slots", n, len(slots))
	}

	// Corrupt the model's q/v in place (zero them) so a faithful reload MUST restore
	// the base values — a no-op reload would pass vacuously otherwise.
	for i, s := range slots {
		z := mlx.MultiplyScalar(base[i], 0) // same shape/dtype, all zeros
		if err := mlx.Eval(z); err != nil {
			t.Fatalf("eval zero: %v", err)
		}
		if err := s.set(z); err != nil {
			t.Fatalf("zero slot %s: %v", s.path, err)
		}
	}

	// Reload the checkpoint and confirm every q/v slot is bit-exact to the base.
	got, err := loadTrainedQV(m, ckpt)
	if err != nil {
		t.Fatalf("loadTrainedQV: %v", err)
	}
	if got != len(slots) {
		t.Fatalf("loadTrainedQV applied %d tensors, want %d", got, len(slots))
	}
	slots2, reloaded, err := collectSlots(m)
	if err != nil {
		t.Fatalf("collectSlots after reload: %v", err)
	}
	var worst float64
	for i, s := range slots2 {
		want, ok := baseByPath[s.path]
		if !ok {
			t.Fatalf("reloaded slot %q not in base", s.path)
		}
		d := maxAbsDiff(t, want, reloaded[i])
		if d > worst {
			worst = d
		}
		if d != 0 {
			t.Errorf("slot %q round-trip not bit-exact: max|Δ|=%g", s.path, d)
		}
	}
	t.Logf("checkpoint round-trip bit-exact over %d q/v tensors (worst max|Δ|=%g)", len(slots2), worst)
}

// TestCKPTBPhase1FitsScore0PlusTrain measures whether PHASE 1 of the proposed
// two-process split — held-out@step0 + train + checkpoint the trained q/v — fits
// in ONE process under the Metal array ceiling. The single-process
// score@0+train+score@final instrument crashes in the FINAL decode (decode #7 of
// 12); this probe omits that final decode to confirm score@0+train alone is green
// and that the q/v checkpoint can be taken (snapshotParams) — the de-risk for the
// split. It does NOT score the final policy (that is Phase 2, a separate process).
func TestCKPTBPhase1FitsScore0PlusTrain(t *testing.T) {
	skipIfNoModel(t)
	ctx := context.Background()

	cfg := Config{
		Prompts:     envInt(t, "VT_PROMPTS", 6),
		K:           envInt(t, "VT_K", 4),
		MaxTokens:   envInt(t, "VT_MAXTOK", 32),
		Temperature: 0.8,
		Steps:       envInt(t, "VT_STEPS", 8),
		LR:          1e-6,
		Seed:        envU64(t, "VT_SEED", 1),
	}

	m, err := Load(ctx, DefaultModelDir())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer func() { m.Close(); reclaim() }()

	// Phase 1a — held-out @ step 0.
	set := HeldoutSet()
	step0, err := scoreHeldout(ctx, m, set, heldoutMaxTokens)
	if err != nil {
		t.Fatalf("scoreHeldout step0: %v", err)
	}
	t.Logf("PHASE1 score@0: Avg@1=%.3f (N=%d)", step0.Acc, step0.N)
	reclaim()

	// Phase 1b — train.
	groups, err := buildGroups(ctx, m, cfg, Seeded)
	if err != nil {
		t.Fatalf("buildGroups: %v", err)
	}
	defer freeGroups(groups)
	mech, err := runMethod(ctx, m, SweepConfigs()[0].Method, cfg, groups)
	if err != nil {
		t.Fatalf("runMethod (C1 baseline): %v", err)
	}
	reclaim()

	// Phase 1c — take the q/v checkpoint (what would cross to Phase 2). This is the
	// exact snapshot restoreParams/SetWeights would re-apply; here we only confirm
	// it materializes without busting the ceiling.
	slots, snap, err := snapshotParams(m)
	if err != nil {
		t.Fatalf("snapshotParams (q/v checkpoint): %v", err)
	}
	defer func() {
		for _, a := range snap {
			a.Free()
		}
	}()
	t.Logf("PHASE1 OK: train steps=%d lossFinite=%v; q/v checkpoint = %d tensors (paths like %q)",
		mech.Steps, mech.LossFinite, len(snap), slots[0].path)
	if len(snap) == 0 {
		t.Fatal("empty q/v checkpoint")
	}
}
