//go:build modelir

package methodcompare

import (
	"math"
	"strings"
	"testing"
)

// The modelir layer fills the model-coupled metrics (FinalLoss, WallMillis) onto
// the deterministic core, without disturbing the core mechanism metrics. The
// model metrics are finite (a real toy forward pass), and the core metrics match
// the tag-free Evaluate exactly — the model layer is additive.
func TestEvaluateWithModelFillsModelMetricsAdditively(t *testing.T) {
	const seed = 5
	core, err := Evaluate(seed)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	full, err := EvaluateWithModel(seed)
	if err != nil {
		t.Fatalf("EvaluateWithModel: %v", err)
	}
	if len(full) != len(core) {
		t.Fatalf("model run has %d rows, core %d", len(full), len(core))
	}
	for i := range full {
		// Core mechanism fields untouched by the model layer.
		if full[i].Method != core[i].Method ||
			full[i].AdvAbsMean != core[i].AdvAbsMean ||
			full[i].AdvStd != core[i].AdvStd ||
			full[i].WMEMean != core[i].WMEMean ||
			full[i].ClipBindHigh != core[i].ClipBindHigh ||
			full[i].AdvVarAcrossSteps != core[i].AdvVarAcrossSteps ||
			full[i].GroupsKept != core[i].GroupsKept ||
			full[i].CliffJSDTerm != core[i].CliffJSDTerm {
			t.Fatalf("method %q: model layer altered a core metric:\n core %+v\n full %+v",
				core[i].Method, core[i], full[i])
		}
		// Model metrics present and finite.
		if full[i].FinalLoss == nil {
			t.Fatalf("method %q: model layer left FinalLoss nil", full[i].Method)
		}
		if math.IsNaN(*full[i].FinalLoss) || math.IsInf(*full[i].FinalLoss, 0) {
			t.Fatalf("method %q: non-finite model loss %v", full[i].Method, *full[i].FinalLoss)
		}
		if full[i].WallMillis < 0 {
			t.Fatalf("method %q: negative wall-time %v", full[i].Method, full[i].WallMillis)
		}
	}
}

// The table-with-model section is clearly labeled non-reproducible, so nobody
// diffs wall-time across runs and reads a regression.
func TestTableWithModelLabelsNonReproducible(t *testing.T) {
	full, err := EvaluateWithModel(2)
	if err != nil {
		t.Fatalf("EvaluateWithModel: %v", err)
	}
	out := TableWithModel(2, full)
	if !strings.Contains(out, "NON-REPRODUCIBLE") {
		t.Fatalf("modelir table section is not labeled NON-REPRODUCIBLE:\n%s", out)
	}
	if !strings.Contains(out, "TOY MECHANISM, not paper accuracy") {
		t.Fatalf("table missing the toy-mechanism honesty header")
	}
}
