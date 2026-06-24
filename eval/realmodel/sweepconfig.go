// Package realmodel's sweep configuration is defined here, tag-free, so the
// config invariants (C1 == zero Method) can be asserted without loading the real
// model. The model-bearing sweep driver (RunSweptConfig) lives in sweep.go behind
// the modelir tag.
package realmodel

import (
	"fmt"
	"math"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// SweptConfig is one config in the differential training sweep (C1..C5). It pairs
// a named method configuration with the same compute knobs (prompts/k/max-tokens/
// steps) and seed every other config runs at, so the only thing distinguishing the
// configs is the method. The held-out probe set is fixed and shared across all of
// them (HeldoutSet), so the per-config Δacc isolates the method's effect.
//
// The five configs are NOT the 9-method toy registry (single-knob rows + a
// kitchen-sink all-on); they are the directive's deliberate compound ladder:
//
//	C1 Baseline   — MGPO, symmetric clip 0.2, λ as DESIGN.
//	C2 Tier-1     — Dr.GRPO debias + DAPO Clip-Higher (0.2/0.28).
//	C3 DCPO-SAS   — Smooth Advantage Standardization across steps.
//	C4 HDPO       — cliff-JSD self-teacher term.
//	C5 Composed   — Tier-1 + DCPO-SAS + HDPO stacked.
type SweptConfig struct {
	Name   string               // C1..C5 display name
	Method methodcompare.Method // the method-comparison configuration
}

// SweepConfigs returns the five swept configs in C1..C5 order. C1's Method is the
// zero Method (baseline), bit-for-bit the same configuration the toy harness and
// the existing real-model "baseline" row use, so the zero-value==baseline
// invariant holds here too. The compound configs are built from the same mgpo
// option surface the registry uses, so the sweep configuration IS the training
// configuration.
func SweepConfigs() []SweptConfig {
	const lambda = 1.0
	return []SweptConfig{
		{Name: "C1-baseline", Method: methodcompare.Method{Name: "C1-baseline", Lambda: lambda}},
		{Name: "C2-tier1", Method: methodcompare.Method{
			Name:       "C2-tier1",
			Lambda:     lambda,
			Opts:       mgpo.Options{DrGRPOAdvantage: true, ClipEpsLow: 0.2, ClipEpsHigh: 0.28},
			DrGRPOLoss: true,
		}},
		{Name: "C3-dcpo-sas", Method: methodcompare.Method{
			Name:          "C3-dcpo-sas",
			Lambda:        lambda,
			DCPOSmoothing: true,
		}},
		{Name: "C4-hdpo", Method: methodcompare.Method{
			Name:   "C4-hdpo",
			Lambda: lambda,
			HDPO:   mgpo.HDPOConfig{LambdaJSD: 0.5},
		}},
		{Name: "C5-composed", Method: methodcompare.Method{
			Name:          "C5-composed",
			Lambda:        lambda,
			Opts:          mgpo.Options{DrGRPOAdvantage: true, ClipEpsLow: 0.2, ClipEpsHigh: 0.28},
			DrGRPOLoss:    true,
			DCPOSmoothing: true,
			HDPO:          mgpo.HDPOConfig{LambdaJSD: 0.5},
		}},
	}
}

// SweepConfigCount is the number of swept configs (C1..C5).
func SweepConfigCount() int { return len(SweepConfigs()) }

// SweepConfigName returns the swept config name at idx (for an ERROR cell when a
// child trial dies before reporting).
func SweepConfigName(idx int) string {
	cfgs := SweepConfigs()
	if idx < 0 || idx >= len(cfgs) {
		return fmt.Sprintf("config#%d", idx)
	}
	return cfgs[idx].Name
}

// DeltaSpread returns the min, max, and (max−min) spread of a set of held-out
// deltas — the noise-floor summary over repeated seeds of one config. Any method
// must beat this spread to count as a real effect, not run-to-run wobble.
func DeltaSpread(deltas []float64) (min, max, spread float64) {
	if len(deltas) == 0 {
		return 0, 0, 0
	}
	min, max = math.Inf(1), math.Inf(-1)
	for _, d := range deltas {
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return min, max, max - min
}

// assertZeroMethodIsBaseline re-asserts the zero-value==baseline invariant for the
// sweep: C1's Method must be the zero Method except for its name and the shared
// lambda — no advantage refinement, no clip override, no DCPO/HDPO/FRPO/DRA. It is
// used by the invariant test and documents the contract at the construction site.
func assertZeroMethodIsBaseline() error {
	c1 := SweepConfigs()[0].Method
	zero := methodcompare.Method{Name: c1.Name, Lambda: c1.Lambda}
	if c1.Opts != zero.Opts {
		return fmt.Errorf("C1 Opts %+v != zero %+v", c1.Opts, zero.Opts)
	}
	if c1.DrGRPOLoss || c1.DCPOSmoothing || c1.DynamicSampling {
		return fmt.Errorf("C1 has a refinement flag set (DrGRPOLoss=%v DCPOSmoothing=%v DynamicSampling=%v)",
			c1.DrGRPOLoss, c1.DCPOSmoothing, c1.DynamicSampling)
	}
	if c1.FRPO.BetaFuture != 0 || c1.HDPO.LambdaJSD != 0 || c1.DRA != nil {
		return fmt.Errorf("C1 has a sibling-loss/reweight set (FRPO=%+v HDPO=%+v DRA!=nil:%v)",
			c1.FRPO, c1.HDPO, c1.DRA != nil)
	}
	return nil
}
