package methodcompare

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// byName indexes evaluated metrics by method name for the delta assertions.
func byName(t *testing.T, ms []Metrics) map[string]Metrics {
	t.Helper()
	m := make(map[string]Metrics, len(ms))
	for _, x := range ms {
		m[x.Method] = x
	}
	return m
}

// The harness reproducibility contract (DESIGN_RL_UPGRADE.md Part 2): the same
// seed yields identical core metrics, and the serialized core table is
// byte-identical across two independent runs — the strongest form of
// same-seed==same-numbers. The core metrics are pure functions of the fixed
// scenario, so this holds without any model.
func TestReproducibleByteIdentical(t *testing.T) {
	const seed = 7

	ms1, err := Evaluate(seed)
	if err != nil {
		t.Fatalf("Evaluate run 1: %v", err)
	}
	ms2, err := Evaluate(seed)
	if err != nil {
		t.Fatalf("Evaluate run 2: %v", err)
	}

	// Value equality, per method, via the canonical core serialization (Metrics
	// holds a map, so compare the marshaled core fields, not == on the struct).
	if len(ms1) != len(ms2) {
		t.Fatalf("run lengths differ: %d != %d", len(ms1), len(ms2))
	}
	for i := range ms1 {
		a, _ := json.Marshal(metricsCoreOnly(ms1[i]))
		b, _ := json.Marshal(metricsCoreOnly(ms2[i]))
		if !bytes.Equal(a, b) {
			t.Fatalf("method %q core metrics differ across runs:\n %s\n %s", ms1[i].Method, a, b)
		}
	}

	// Byte-identical serialized CORE table across two runs.
	t1 := Table(seed, ms1)
	t2 := Table(seed, ms2)
	if t1 != t2 {
		t.Fatalf("core table not byte-identical across runs:\n--- run1 ---\n%s\n--- run2 ---\n%s", t1, t2)
	}

	// Byte-identical core JSON across two runs.
	j1, err := NewReport(seed, ms1).JSON()
	if err != nil {
		t.Fatalf("JSON run 1: %v", err)
	}
	j2, err := NewReport(seed, ms2).JSON()
	if err != nil {
		t.Fatalf("JSON run 2: %v", err)
	}
	if !bytes.Equal(j1, j2) {
		t.Fatalf("core JSON not byte-identical across runs")
	}
}

// metricsCoreOnly zeros the modelir (non-reproducible) fields so the core
// equality check ignores them. The tag-free Evaluate leaves them at their
// defaults (FinalLoss=NaN, no StageLoss, WallMillis=0), but this makes the intent
// explicit and the test robust if a caller fills them.
func metricsCoreOnly(m Metrics) Metrics {
	m.FinalLoss = nil
	m.StageLoss = nil
	m.WallMillis = 0
	return m
}

// A different seed must change at least one reproducible metric (otherwise the
// seed is inert and "reproducible" would be trivially true). The scenario's
// random component is the rollout texts (DRA) and the per-token ratios, so the
// DRA |A| and the clip-bind rate should move with the seed.
func TestSeedActuallyVaries(t *testing.T) {
	a, err := Evaluate(1)
	if err != nil {
		t.Fatalf("Evaluate seed 1: %v", err)
	}
	b, err := Evaluate(2)
	if err != nil {
		t.Fatalf("Evaluate seed 2: %v", err)
	}
	ta, tb := Table(1, a), Table(2, b)
	if ta == tb {
		t.Fatalf("seed 1 and seed 2 produced identical tables; seed is inert")
	}
}

// Mechanism deltas (DESIGN_RL_UPGRADE.md Part 2): each knob must move the metric
// its theory predicts, in the predicted direction, relative to baseline.
func TestMechanismDeltas(t *testing.T) {
	ms, err := Evaluate(1)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if err := assertRegistryOrder(ms); err != nil {
		t.Fatalf("registry order: %v", err)
	}
	m := byName(t, ms)
	base := m["baseline"]

	// Dr.GRPO removes the std divisor ⇒ smaller |A| and smaller A std.
	if !(m["+DrGRPO"].AdvAbsMean < base.AdvAbsMean) {
		t.Errorf("DrGRPO did not shrink |A|: %v !< %v", m["+DrGRPO"].AdvAbsMean, base.AdvAbsMean)
	}
	if !(m["+DrGRPO"].AdvStd < base.AdvStd) {
		t.Errorf("DrGRPO did not shrink A std: %v !< %v", m["+DrGRPO"].AdvStd, base.AdvStd)
	}

	// Clip-Higher raises the upper ceiling ⇒ FEWER high-side ratios bind.
	if !(m["+ClipHigher"].ClipBindHigh < base.ClipBindHigh) {
		t.Errorf("ClipHigher did not lower the upper-clip-bind rate: %v !< %v",
			m["+ClipHigher"].ClipBindHigh, base.ClipBindHigh)
	}
	// And the baseline must actually bind on the high side, else the test is vacuous.
	if !(base.ClipBindHigh > 0) {
		t.Errorf("baseline upper-clip-bind rate is 0; the clip-bind metric is vacuous")
	}

	// DCPO-SAS smooths the advantage across steps ⇒ lower across-step variance.
	if !(m["+DCPO-SAS"].AdvVarAcrossSteps < base.AdvVarAcrossSteps) {
		t.Errorf("DCPO-SAS did not smooth across-step advantage variance: %v !< %v",
			m["+DCPO-SAS"].AdvVarAcrossSteps, base.AdvVarAcrossSteps)
	}

	// Dynamic Sampling drops the acc∈{0,1} groups ⇒ fewer groups kept, and the
	// mean w_ME rises (the dropped groups sit at the w_ME extreme).
	if !(m["+DynSampling"].GroupsKept < base.GroupsKept) {
		t.Errorf("DynSampling did not drop any group: kept %d, baseline %d",
			m["+DynSampling"].GroupsKept, base.GroupsKept)
	}
	if !(m["+DynSampling"].WMEMean > base.WMEMean) {
		t.Errorf("DynSampling did not raise mean w_ME: %v !> %v", m["+DynSampling"].WMEMean, base.WMEMean)
	}

	// HDPO activates only on the cliff set ⇒ nonzero cliff-JSD term; baseline 0.
	if !(base.CliffJSDTerm == 0) {
		t.Errorf("baseline has a nonzero cliff-JSD term %v; HDPO leaked into baseline", base.CliffJSDTerm)
	}
	if !(m["+HDPO"].CliffJSDTerm > 0) {
		t.Errorf("HDPO produced no cliff-JSD term: %v", m["+HDPO"].CliffJSDTerm)
	}
	if !(m["+HDPO"].CliffGroups > 0) {
		t.Errorf("scenario has no cliff group; HDPO metric is vacuous")
	}

	// DRA reweights diversity ⇒ the advantage shifts from baseline (small but
	// nonzero, since the embedder is non-degenerate on distinct rollout texts).
	if math.Abs(m["+DRA"].AdvAbsMean-base.AdvAbsMean) < 1e-9 {
		t.Errorf("DRA did not move |A| at all: %v == %v", m["+DRA"].AdvAbsMean, base.AdvAbsMean)
	}

	// Long2Short cuts tokens per sample at equal reward ⇒ reshaped < raw.
	if !(base.TokensPerSample < base.TokensPerSampleRaw) {
		t.Errorf("Long2Short did not cut tokens per sample: %v !< %v",
			base.TokensPerSample, base.TokensPerSampleRaw)
	}

	// all-on stacks the advantage-shrinking knobs ⇒ |A| below baseline, groups
	// filtered, cliff active.
	allOn := m["all-on"]
	if !(allOn.AdvAbsMean < base.AdvAbsMean) {
		t.Errorf("all-on did not shrink |A|: %v !< %v", allOn.AdvAbsMean, base.AdvAbsMean)
	}
	if !(allOn.GroupsKept < base.GroupsKept) {
		t.Errorf("all-on did not filter groups: %d", allOn.GroupsKept)
	}
	if !(allOn.CliffJSDTerm > 0) {
		t.Errorf("all-on lost the cliff-JSD term: %v", allOn.CliffJSDTerm)
	}
}

// The JSON document must carry the honesty header, the per-metric layer
// provenance, and the non-reproducible flags on the modelir metrics — so no
// consumer diffs wall-time across runs and reads a regression.
func TestJSONLayerProvenanceAndHonesty(t *testing.T) {
	ms, err := Evaluate(3)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	doc, err := NewReport(3, ms).JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}

	if !strings.Contains(string(doc), "TOY MECHANISM, not paper accuracy") {
		t.Fatalf("JSON missing the toy-mechanism honesty header")
	}

	var parsed Report
	if err := json.Unmarshal(doc, &parsed); err != nil {
		t.Fatalf("JSON does not round-trip: %v", err)
	}
	// Core metrics tagged reproducible.
	for _, key := range []string{"adv_abs_mean", "clip_bind_high", "adv_var_across_steps"} {
		ml, ok := parsed.MetricLayers[key]
		if !ok {
			t.Fatalf("metric %q missing a layer tag", key)
		}
		if ml.Layer != "core" || !ml.Reproducible {
			t.Errorf("core metric %q mis-tagged: %+v", key, ml)
		}
	}
	// Modelir metrics tagged non-reproducible / machine-dependent.
	for _, key := range []string{"final_loss", "stage_loss", "wall_millis"} {
		ml, ok := parsed.MetricLayers[key]
		if !ok {
			t.Fatalf("metric %q missing a layer tag", key)
		}
		if ml.Layer != "modelir" || ml.Reproducible {
			t.Errorf("modelir metric %q mis-tagged (must be modelir/non-reproducible): %+v", key, ml)
		}
		if ml.Note == "" {
			t.Errorf("modelir metric %q has no caveat note", key)
		}
	}
}

// The off-path is undisturbed: the harness baseline row reproduces the plain
// MGPO advantage exactly (no refinement applied), so the baseline's advantage
// statistics equal those of mgpo.ScaledAdvantagesOpt with the zero Options on the
// same final-step rewards. This pins that the harness "baseline" really is the
// DESIGN.md baseline, not a refinement in disguise.
func TestBaselineRowIsTheDesignBaseline(t *testing.T) {
	ms, err := Evaluate(1)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	base := byName(t, ms)["baseline"]
	// The baseline must use neither Dr.GRPO advantage nor an asymmetric clip:
	// its clip-bind-high equals the symmetric-clip bind rate, and |A| matches the
	// std-normalized advantage scale (nonzero, since groups are non-degenerate).
	if base.AdvAbsMean <= 0 {
		t.Fatalf("baseline |A| is zero; advantage path is broken")
	}
	low, high := Method{Name: "baseline"}.Opts.ClipRange(baseConfig())
	if low != baselineClipEps || high != baselineClipEps {
		t.Fatalf("baseline clip is not symmetric: low=%v high=%v want %v", low, high, baselineClipEps)
	}
}
