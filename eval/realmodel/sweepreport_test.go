//go:build modelir

package realmodel

import (
	"strings"
	"testing"
)

// These tests exercise the sweep report's pure stitch/rank/recommend logic. They
// are modelir-tagged only because StitchedResult/Metrics live behind that tag —
// they load NO model and run in milliseconds.

func mkResult(name string, acc0, accFinal float64) StitchedResult {
	return StitchedResult{
		Config: name, Source: "SEEDED", AccStep0: acc0, AccFinal: accFinal,
		DeltaAcc: accFinal - acc0, HeldoutN: 12, StepsRun: 8,
		Mechanism: Metrics{Method: name, Source: "SEEDED", LossFinite: true},
	}
}

func TestStitchMergesPhases(t *testing.T) {
	p1 := SweepRow{Status: "ok", Phase: 1, Index: 0, Config: "C1-baseline", Source: "SEEDED", Seed: 1,
		Acc: 0.5, HeldoutN: 12, StepsRun: 8, CkptPath: "/tmp/x", WallMillis: 1000,
		Mechanism: Metrics{Method: "C1-baseline", RatioVar: 0.003, LossFinite: true}}
	p2 := SweepRow{Status: "ok", Phase: 2, Index: 0, Config: "C1-baseline", Source: "SEEDED", Seed: 1,
		Acc: 0.583, HeldoutN: 12, WallMillis: 500}
	r := Stitch(p1, p2, "", "")
	if r.AccStep0 != 0.5 || r.AccFinal != 0.583 {
		t.Fatalf("acc0=%v accFinal=%v", r.AccStep0, r.AccFinal)
	}
	if d := r.DeltaAcc - 0.083; d > 1e-9 || d < -1e-9 {
		t.Fatalf("delta=%v want ~0.083", r.DeltaAcc)
	}
	if r.StepsRun != 8 || r.Mechanism.RatioVar != 0.003 {
		t.Fatalf("stepsRun/mech not carried from P1: %+v", r)
	}
	if r.WallMillis != 1500 {
		t.Fatalf("wall not summed: %v", r.WallMillis)
	}
	if r.ErrMsg != "" {
		t.Fatalf("unexpected err: %s", r.ErrMsg)
	}
}

func TestStitchPhaseErrorIsErrorCell(t *testing.T) {
	p1 := SweepRow{Status: "error", Phase: 1, Index: 2, Config: "C3-dcpo-sas", Source: "SEEDED", Seed: 1}
	r := Stitch(p1, SweepRow{}, "phase 1 child exited: signal: killed", "phase 2 skipped")
	if r.ErrMsg == "" {
		t.Fatal("want ERROR cell")
	}
	if r.Config != "C3-dcpo-sas" || r.Source != "SEEDED" {
		t.Fatalf("identity lost on error: %+v", r)
	}
	if r.Mechanism.LossFinite {
		t.Fatal("error cell mechanism should be LossFinite=false")
	}
}

func TestRankResultsByDeltaErrorsLast(t *testing.T) {
	results := []StitchedResult{
		mkResult("C1-baseline", 0.5, 0.5),   // Δ 0.0
		mkResult("C2-tier1", 0.5, 0.583),    // Δ +0.083
		mkResult("C3-dcpo-sas", 0.5, 0.417), // Δ -0.083
		{Config: "C4-hdpo", Source: "SEEDED", ErrMsg: "boom", Mechanism: ErrorMetrics("C4-hdpo", "SEEDED")},
		mkResult("C5-composed", 0.5, 0.583), // Δ +0.083 (ties C2; name breaks tie)
	}
	got := RankResults(results)
	want := []string{"C2-tier1", "C5-composed", "C1-baseline", "C3-dcpo-sas", "C4-hdpo"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("ranking = %v, want %v", got, want)
	}
}

func TestRecommendIndistinguishableWhenBelowNoiseFloor(t *testing.T) {
	results := []StitchedResult{
		mkResult("C1-baseline", 0.5, 0.5),
		mkResult("C2-tier1", 0.5, 0.583), // +0.083
	}
	ranking := RankResults(results)
	nf := &NoiseFloor{Config: "C1-baseline", Seeds: []uint64{1, 2}, Deltas: []float64{0.0, 0.167}, Spread: 0.167}
	rec := recommend(results, ranking, nf)
	if !strings.Contains(rec, "INDISTINGUISHABLE") {
		t.Fatalf("want INDISTINGUISHABLE (best Δ 0.083 < noise spread 0.167), got: %s", rec)
	}
	if !strings.Contains(rec, "C1-baseline") {
		t.Fatalf("indistinguishable verdict should default to C1-baseline: %s", rec)
	}
}

func TestRecommendNamesWinnerWhenAboveNoiseFloor(t *testing.T) {
	results := []StitchedResult{
		mkResult("C1-baseline", 0.5, 0.5),
		mkResult("C2-tier1", 0.5, 0.75), // +0.25
	}
	ranking := RankResults(results)
	nf := &NoiseFloor{Config: "C1-baseline", Seeds: []uint64{1, 2}, Deltas: []float64{0.0, 0.083}, Spread: 0.083}
	rec := recommend(results, ranking, nf)
	if !strings.Contains(rec, "RECOMMEND C2-tier1") {
		t.Fatalf("want RECOMMEND C2-tier1 (Δ 0.25 > noise 0.083), got: %s", rec)
	}
}

func TestSweepReportTableAndJSONRender(t *testing.T) {
	results := []StitchedResult{
		mkResult("C1-baseline", 0.5, 0.5),
		mkResult("C2-tier1", 0.5, 0.583),
	}
	rep := NewSweepReport("/models/x", DefaultConfig(), results, nil)
	tbl := rep.SweepTable()
	for _, want := range []string{"DIRECTIONAL TRAINING SWEEP", "RANKING", "RECOMMENDATION", "NOISE FLOOR"} {
		if !strings.Contains(tbl, want) {
			t.Errorf("table missing %q", want)
		}
	}
	doc, err := rep.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	for _, want := range []string{"\"delta_acc\"", "\"ranking\"", "\"recommendation\"", "\"non_reproducible\""} {
		if !strings.Contains(string(doc), want) {
			t.Errorf("JSON missing %q", want)
		}
	}
}
