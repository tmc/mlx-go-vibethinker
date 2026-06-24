package realmodel

import (
	"math"
	"testing"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// TestSweepC1IsZeroBaseline re-asserts the zero-value==baseline invariant at the
// sweep level: C1 must be the zero Method (no advantage refinement, no clip
// override, no DCPO/HDPO/FRPO/DRA) so it is a true baseline anchor against which
// the Δacc of C2..C5 is read. It runs tag-free — no model — alongside the
// signal/mgpo bit-identical invariant tests it depends on.
func TestSweepC1IsZeroBaseline(t *testing.T) {
	if err := assertZeroMethodIsBaseline(); err != nil {
		t.Fatalf("C1 is not the zero baseline: %v", err)
	}
}

// TestSweepConfigsAreTheDirectiveLadder pins the five swept configs to the
// directive's compound ladder (C1..C5) by their distinguishing knobs, so an
// accidental edit to a config that silently changes what the sweep ranks is
// caught here, not in a four-minute model run.
func TestSweepConfigsAreTheDirectiveLadder(t *testing.T) {
	cfgs := SweepConfigs()
	if len(cfgs) != 5 {
		t.Fatalf("want 5 swept configs (C1..C5), got %d", len(cfgs))
	}
	tier1Opts := mgpo.Options{DrGRPOAdvantage: true, ClipEpsLow: 0.2, ClipEpsHigh: 0.28}

	// C1 baseline: zero Method (covered by assertZeroMethodIsBaseline too).
	if c := cfgs[0]; c.Method.DrGRPOLoss || c.Method.Opts != (mgpo.Options{}) || c.Method.HDPO.LambdaJSD != 0 || c.Method.DCPOSmoothing {
		t.Errorf("C1 baseline has a refinement set: %+v", c.Method)
	}
	// C2 Tier-1: Dr.GRPO debias (advantage + loss) + Clip-Higher 0.2/0.28, nothing else.
	if c := cfgs[1].Method; c.Opts != tier1Opts || !c.DrGRPOLoss || c.DCPOSmoothing || c.HDPO.LambdaJSD != 0 {
		t.Errorf("C2 tier1 not {Dr.GRPO + ClipHigher} only: %+v", c)
	}
	// C3 DCPO-SAS: smoothing only.
	if c := cfgs[2].Method; !c.DCPOSmoothing || c.DrGRPOLoss || c.Opts != (mgpo.Options{}) || c.HDPO.LambdaJSD != 0 {
		t.Errorf("C3 not {DCPO-SAS} only: %+v", c)
	}
	// C4 HDPO: cliff-JSD only.
	if c := cfgs[3].Method; c.HDPO.LambdaJSD == 0 || c.DCPOSmoothing || c.DrGRPOLoss || c.Opts != (mgpo.Options{}) {
		t.Errorf("C4 not {HDPO} only: %+v", c)
	}
	// C5 Composed: Tier-1 + DCPO-SAS + HDPO stacked.
	if c := cfgs[4].Method; c.Opts != tier1Opts || !c.DrGRPOLoss || !c.DCPOSmoothing || c.HDPO.LambdaJSD == 0 {
		t.Errorf("C5 composed not {Tier-1 + DCPO-SAS + HDPO}: %+v", c)
	}

	// The directive forbids adding SDPO/SRPO/QAE/GSPO/GMPO/VAPO: every config must
	// be built only from the mgpo knobs the ladder names. DynamicSampling and DRA
	// and FRPO are deliberately absent from the sweep ladder (C1..C5).
	for _, c := range cfgs {
		if c.Method.DynamicSampling || c.Method.DRA != nil || c.Method.FRPO.BetaFuture != 0 {
			t.Errorf("%s carries a knob outside the C1..C5 ladder (DynSampling/DRA/FRPO): %+v", c.Name, c.Method)
		}
	}
}

// TestSweepConfigNameAndCount guards the helpers the subprocess harness uses to
// place ERROR cells.
func TestSweepConfigNameAndCount(t *testing.T) {
	if SweepConfigCount() != 5 {
		t.Fatalf("SweepConfigCount() = %d, want 5", SweepConfigCount())
	}
	if got := SweepConfigName(0); got != "C1-baseline" {
		t.Errorf("SweepConfigName(0) = %q, want C1-baseline", got)
	}
	if got := SweepConfigName(-1); got != "config#-1" {
		t.Errorf("SweepConfigName(-1) = %q, want config#-1", got)
	}
	if got := SweepConfigName(99); got != "config#99" {
		t.Errorf("SweepConfigName(99) = %q, want config#99", got)
	}
}

// TestDeltaSpread checks the noise-floor summary helper: the spread is max−min
// over the per-seed deltas, and an empty slice is a zero spread (no data).
func TestDeltaSpread(t *testing.T) {
	min, max, spread := DeltaSpread([]float64{0.05, -0.02, 0.11, 0.0})
	if math.Abs(min-(-0.02)) > 1e-12 || math.Abs(max-0.11) > 1e-12 || math.Abs(spread-0.13) > 1e-12 {
		t.Fatalf("DeltaSpread = (min %.4f, max %.4f, spread %.4f), want (-0.02, 0.11, 0.13)", min, max, spread)
	}
	if _, _, s := DeltaSpread(nil); s != 0 {
		t.Fatalf("DeltaSpread(nil) spread = %.4f, want 0", s)
	}
}

// Ensure methodcompare is referenced even if the assertions above are edited.
var _ = methodcompare.Method{}
