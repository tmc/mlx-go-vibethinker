package methodcompare

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

// reportHeader is the honesty banner printed atop every table and embedded in
// every JSON document. It states plainly that the numbers measure mechanism on a
// toy substrate, not paper/benchmark accuracy.
const reportHeader = "method-comparison: TOY MECHANISM, not paper accuracy " +
	"(synthetic-scenario deltas show whether each knob moves the metric its theory predicts; " +
	"they are NOT benchmark accuracy deltas)"

// metricLayer records, per metric field, which layer produced it and whether it
// is reproducible (bit-stable for a fixed seed). The core mechanism metrics are
// pure functions of the fixed scenario and are reproducible; the modelir metrics
// (per-stage loss, wall-time) are model- and machine-dependent and are NOT
// reproducible across runs.
type metricLayer struct {
	Layer        string `json:"layer"`        // "core" or "modelir"
	Reproducible bool   `json:"reproducible"` // true iff bit-stable for a fixed seed
	Note         string `json:"note,omitempty"`
}

// metricLayers maps each Metrics JSON field to its provenance. Anything not
// listed defaults to core/reproducible (the mechanism metrics).
func metricLayers() map[string]metricLayer {
	core := metricLayer{Layer: "core", Reproducible: true}
	return map[string]metricLayer{
		"adv_mean":             core,
		"adv_std":              core,
		"adv_abs_mean":         core,
		"wme_mean":             core,
		"wme_min":              core,
		"wme_max":              core,
		"clip_bind_rate":       core,
		"clip_bind_high":       core,
		"tokens_per_sample":    core,
		"adv_var_across_steps": core,
		"groups_kept":          core,
		"groups_in":            core,
		"cliff_groups":         core,
		"cliff_jsd_term":       core,
		"final_loss": {Layer: "modelir", Reproducible: false,
			Note: "toy-model forward pass; depends on the model build and is not bit-reproducible"},
		"stage_loss": {Layer: "modelir", Reproducible: false,
			Note: "per-stage toy-pipeline loss; model-dependent, not bit-reproducible"},
		"wall_millis": {Layer: "modelir", Reproducible: false,
			Note: "machine-dependent wall-time; never diff across runs or machines to infer a regression"},
	}
}

// Report is the full machine-readable comparison document.
type Report struct {
	Header       string                 `json:"header"`
	Seed         uint64                 `json:"seed"`
	MetricLayers map[string]metricLayer `json:"metric_layers"`
	Methods      []Metrics              `json:"methods"`
}

// NewReport assembles a Report from evaluated metrics.
func NewReport(seed uint64, methods []Metrics) Report {
	return Report{
		Header:       reportHeader,
		Seed:         seed,
		MetricLayers: metricLayers(),
		Methods:      methods,
	}
}

// JSON serializes the report deterministically (sorted map keys via the encoder,
// fixed field order via the structs), so the same metrics always produce
// byte-identical output. It returns indented JSON with a trailing newline.
func (r Report) JSON() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, fmt.Errorf("methodcompare: marshal report: %w", err)
	}
	return buf.Bytes(), nil
}

// Table renders the core mechanism metrics as a fixed-width comparable table,
// one row per method in registry order, with the honesty header on top. Only the
// reproducible core metrics are shown; the modelir loss/wall-time, when present,
// are appended in a separate clearly-labeled section by TableWithModel.
func Table(seed uint64, methods []Metrics) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "%s\nseed=%d\n\n", reportHeader, seed)

	cols := []struct {
		head string
		get  func(Metrics) string
	}{
		{"method", func(m Metrics) string { return m.Method }},
		{"|A|mean", func(m Metrics) string { return f(m.AdvAbsMean) }},
		{"A std", func(m Metrics) string { return f(m.AdvStd) }},
		{"wME mean", func(m Metrics) string { return f(m.WMEMean) }},
		{"clipBind", func(m Metrics) string { return f(m.ClipBindRate) }},
		{"clipHi", func(m Metrics) string { return f(m.ClipBindHigh) }},
		{"tokRaw", func(m Metrics) string { return f(m.TokensPerSampleRaw) }},
		{"tokL2S", func(m Metrics) string { return f(m.TokensPerSample) }},
		{"advVarΔstp", func(m Metrics) string { return f(m.AdvVarAcrossSteps) }},
		{"keep/in", func(m Metrics) string { return fmt.Sprintf("%d/%d", m.GroupsKept, m.GroupsIn) }},
		{"cliffJSD", func(m Metrics) string { return f(m.CliffJSDTerm) }},
	}

	widths := make([]int, len(cols))
	for i, c := range cols {
		widths[i] = len(c.head)
	}
	rows := make([][]string, len(methods))
	for r, m := range methods {
		row := make([]string, len(cols))
		for i, c := range cols {
			row[i] = c.get(m)
			if len(row[i]) > widths[i] {
				widths[i] = len(row[i])
			}
		}
		rows[r] = row
	}

	writeRow := func(cells []string) {
		for i, cell := range cells {
			if i > 0 {
				b.WriteString("  ")
			}
			fmt.Fprintf(&b, "%-*s", widths[i], cell)
		}
		b.WriteByte('\n')
	}
	head := make([]string, len(cols))
	for i, c := range cols {
		head[i] = c.head
	}
	writeRow(head)
	sep := make([]string, len(cols))
	for i := range sep {
		sep[i] = repeat('-', widths[i])
	}
	writeRow(sep)
	for _, row := range rows {
		writeRow(row)
	}

	b.WriteString("\nlegend: |A|mean,A std = advantage magnitude/spread (DrGRPO removes the std divisor ⇒ both shrink); " +
		"wME mean = MaxEnt weight (DynSampling drops acc∈{0,1} groups ⇒ mean rises); " +
		"clipBind/clipHi = fraction of token ratios clipped (any/upper) (ClipHigher raises the ceiling ⇒ clipHi falls); " +
		"tokRaw→tokL2S = correct-trace length before/after Long2Short (reshape cuts tokens at equal reward); " +
		"advVarΔstp = advantage variance across steps (DCPO-SAS smooths ⇒ falls); " +
		"keep/in = groups surviving Dynamic Sampling; cliffJSD = HDPO term (nonzero only on the cliff set).\n" +
		"all columns are CORE (reproducible) metrics. model loss/wall-time are modelir-only and machine-dependent.\n")
	return b.String()
}

// TableWithModel renders the core table plus a labeled modelir section listing
// the model-coupled, non-reproducible metrics (final loss, wall-time). It is
// used by the modelir layer; the tag-free table omits this section.
func TableWithModel(seed uint64, methods []Metrics) string {
	b := Table(seed, methods)
	var m bytes.Buffer
	m.WriteString("\nmodelir layer (NON-REPRODUCIBLE — model/machine dependent, do not diff across runs):\n")
	fmt.Fprintf(&m, "%-14s  %-12s  %-12s\n", "method", "finalLoss", "wallMillis")
	fmt.Fprintf(&m, "%-14s  %-12s  %-12s\n", repeat('-', 14), repeat('-', 12), repeat('-', 12))
	for _, mt := range methods {
		loss := "n/a"
		if mt.FinalLoss != nil {
			loss = f(*mt.FinalLoss)
		}
		fmt.Fprintf(&m, "%-14s  %-12s  %-12.3f\n", mt.Method, loss, mt.WallMillis)
		for _, k := range stableKeys(mt.StageLoss) {
			fmt.Fprintf(&m, "    %-26s %s\n", k, f(mt.StageLoss[k]))
		}
	}
	return b + m.String()
}

func f(x float64) string {
	if math.IsNaN(x) {
		return "n/a"
	}
	return fmt.Sprintf("%.4f", x)
}

func repeat(c byte, n int) string {
	if n < 0 {
		n = 0
	}
	bs := make([]byte, n)
	for i := range bs {
		bs[i] = c
	}
	return string(bs)
}

// assertRegistryOrder is a guard used by tests: the metrics rows must be in
// registry order so the table and JSON line methods up identically.
func assertRegistryOrder(methods []Metrics) error {
	want := sortedMethodNames()
	if len(methods) != len(want) {
		return fmt.Errorf("methodcompare: %d metrics rows, want %d", len(methods), len(want))
	}
	for i := range want {
		if methods[i].Method != want[i] {
			return fmt.Errorf("methodcompare: row %d is %q, want %q (registry order)", i, methods[i].Method, want[i])
		}
	}
	return nil
}
