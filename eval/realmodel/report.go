//go:build modelir

package realmodel

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// reportHeader is the honesty banner printed atop the table and embedded in the
// JSON. Unlike the toy harness, EVERY number here is model- and machine-
// dependent and NOT reproducible: a real model plus a real optimizer is not
// bit-deterministic across runs. The smoke validates that each method's
// mechanism holds on real logits and that training stays stable — not benchmark
// accuracy (reproducing VibeThinker's published numbers is out of scope).
func reportHeader(steps int) string {
	return fmt.Sprintf("REAL-MODEL MECHANISM SMOKE (Qwen2.5-Math-1.5B, ~%d steps/method); "+
		"subprocess-per-method (VG-graph isolation); mechanism + stability on real logits, "+
		"NOT benchmark accuracy. Every number is model- and machine-dependent and NOT "+
		"reproducible across runs (the tag-free toy harness eval/methodcompare remains the "+
		"byte-identical-repro one).", steps)
}

// Report is the machine-readable real-model comparison document. The
// NonReproducible flag is always true and the note explains why; there is no
// reproducible core here (every metric rides on the real model + optimizer).
// Methods carries the organic block followed by the seeded block, each row
// tagged by its Source. Errors maps a "<SOURCE>/<method>" cell to the reason its
// subprocess trial failed; a present entry means that cell is an ERROR, surfaced
// explicitly (never a silent omission).
type Report struct {
	Schema          int               `json:"schema"`
	Header          string            `json:"header"`
	Model           string            `json:"model"`
	Harness         string            `json:"harness"`
	NonReproducible bool              `json:"non_reproducible"`
	Note            string            `json:"note"`
	MetalCapNote    string            `json:"metal_cap_note"`
	Config          Config            `json:"config"`
	Organic         []Metrics         `json:"organic"`
	Seeded          []Metrics         `json:"seeded"`
	Errors          map[string]string `json:"errors"`
}

const reportNote = "All metrics are real-model + real-optimizer measurements and are NOT bit-reproducible; " +
	"wall-time is machine-dependent. The ratio_* columns are the direct evidence the rescore is live " +
	"(current rescored from post-step weights vs frozen old): ratio_var>0 and ratio_max_abs_dev>0 mean " +
	"the importance ratio is a real distribution, not a delta at 1. ORGANIC = honest model-generated " +
	"rollouts (a weak base scores ~0%, collapsing within-group reward spread). SEEDED = fixed, " +
	"real-tokenized, real-Forward-rescored completions with guaranteed mixed correctness so the " +
	"reward-shape mechanisms (Dr.GRPO/DCPO/Dynamic Sampling) are observable; the completion TEXT is " +
	"fixed but rescore+loss+optimizer step are real — SEEDED is NOT model accuracy. Each (method x source) " +
	"cell is computed in its OWN subprocess (VG-graph isolation); a failed trial is an explicit ERROR cell."

// NewReport builds the report document from the run config and the organic +
// seeded per-method metrics, with no trial errors.
func NewReport(model string, cfg Config, organic, seeded []Metrics) Report {
	return NewReportWithErrors(model, cfg, organic, seeded, nil)
}

// NewReportWithErrors builds the report document including the per-cell trial
// errors from the subprocess harness.
func NewReportWithErrors(model string, cfg Config, organic, seeded []Metrics, errs map[string]string) Report {
	if errs == nil {
		errs = map[string]string{}
	}
	return Report{
		Schema:          RowSchema,
		Header:          reportHeader(cfg.Steps),
		Model:           model,
		Harness:         "subprocess-per-method (VG-graph isolation)",
		NonReproducible: true,
		Note:            reportNote,
		MetalCapNote:    metalCapNote(cfg),
		Config:          cfg,
		Organic:         organic,
		Seeded:          seeded,
		Errors:          errs,
	}
}

// metalCapNote states the rollout bound and why it is there, so the small smoke
// config is never read as the achievable maximum.
func metalCapNote(cfg Config) string {
	return fmt.Sprintf("bounded to %d generated tokens x K=%d rollouts x %d prompts to stay under the Metal "+
		"per-process array/command-buffer ceiling (~499000) seen on longer rollouts; this is a mechanism "+
		"smoke, NOT a convergence run, and the bound is not the achievable maximum.",
		cfg.MaxTokens, cfg.K, cfg.Prompts)
}

// JSON renders the report as deterministic (sorted-key) indented JSON.
func (r Report) JSON() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, fmt.Errorf("realmodel: encode report: %w", err)
	}
	return buf.Bytes(), nil
}

// Table renders the human-readable comparison with no trial errors.
func Table(model string, cfg Config, organic, seeded []Metrics) string {
	return TableWithErrors(model, cfg, organic, seeded, nil)
}

// TableWithErrors renders the human-readable comparison: the honesty header, the
// Metal-cap note, then the ORGANIC block followed by the SEEDED block. The
// importance-ratio columns are FIRST-CLASS in both — they are the direct
// evidence the rescore is real, not collapsed to 1. A cell whose trial failed
// (present in errs keyed "<SOURCE>/<method>") is rendered as an explicit ERROR
// row and listed in a trailing errors section, never silently omitted.
func TableWithErrors(model string, cfg Config, organic, seeded []Metrics, errs map[string]string) string {
	var b strings.Builder
	b.WriteString(reportHeader(cfg.Steps))
	b.WriteString("\n\n")
	fmt.Fprintf(&b, "model: %s\nconfig: prompts=%d K=%d maxTok=%d temp=%.2f steps=%d lr=%.0e seed=%d\n",
		model, cfg.Prompts, cfg.K, cfg.MaxTokens, cfg.Temperature, cfg.Steps, cfg.LR, cfg.Seed)
	fmt.Fprintf(&b, "CAP: %s\n\n", metalCapNote(cfg))

	writeBlock(&b, "ORGANIC", "ORGANIC — honest model-generated rollouts (weak base ~0%% accuracy collapses within-group reward spread)", organic, errs)
	b.WriteString("\n")
	writeBlock(&b, "SEEDED", "SEEDED — fixed real-tokenized completions, guaranteed mixed correctness (NOT model accuracy; exhibits reward-shape mechanisms on real logits)", seeded, errs)

	if len(errs) > 0 {
		b.WriteString("\n!!! TRIAL ERRORS (cells marked ERROR above) !!!\n")
		for _, k := range sortedKeys(errs) {
			fmt.Fprintf(&b, "  %s: %s\n", k, errs[k])
		}
	}
	b.WriteString("\nNOTE: NON-REPRODUCIBLE — real model + real optimizer; never diff these across runs/machines.\n")
	return b.String()
}

// writeBlock renders one source block: its label and the three column groups.
// A method whose cell errored (errs["<source>/<method>"] present) is shown as an
// ERROR row in each group.
func writeBlock(b *strings.Builder, source, label string, metrics []Metrics, errs map[string]string) {
	failed := func(method string) bool {
		_, bad := errs[source+"/"+method]
		return bad
	}
	fmt.Fprintf(b, "=== %s ===\n\n", label)

	b.WriteString("  importance-ratio evidence (rescore live iff var>0/maxAbsDev>0; ratio=current/old over masked tokens)\n")
	fmt.Fprintf(b, "  %-14s %10s %10s %12s %12s\n", "method", "ratioMean", "ratioVar", "maxAbsDev", "clipBindHi")
	b.WriteString("  " + repeat("-", 62) + "\n")
	for _, m := range metrics {
		if failed(m.Method) {
			fmt.Fprintf(b, "  %-14s %10s %10s %12s %12s\n", m.Method, "ERROR", "ERROR", "ERROR", "ERROR")
			continue
		}
		fmt.Fprintf(b, "  %-14s %10s %10s %12s %12s\n",
			m.Method, f(m.RatioMean), f(m.RatioVar), f(m.RatioMaxAbsDev), f(m.ClipBindHigh))
	}
	b.WriteString("\n")

	b.WriteString("  advantage / group structure (real rewards)\n")
	fmt.Fprintf(b, "  %-14s %10s %10s %8s %7s %7s %7s\n", "method", "advAbsMean", "advStd", "acc", "cliff", "learn", "drop")
	b.WriteString("  " + repeat("-", 70) + "\n")
	for _, m := range metrics {
		if failed(m.Method) {
			fmt.Fprintf(b, "  %-14s %10s %10s %8s %7s %7s %7s\n", m.Method, "ERROR", "ERROR", "ERROR", "ERR", "ERR", "ERR")
			continue
		}
		fmt.Fprintf(b, "  %-14s %10s %10s %8s %7d %7d %7d\n",
			m.Method, f(m.AdvAbsMean), f(m.AdvStd), f(m.AccMean), m.CliffGroups, m.LearnGroups, m.GroupsDropped)
	}
	b.WriteString("\n")

	b.WriteString("  method terms + stability\n")
	fmt.Fprintf(b, "  %-14s %9s %10s %10s %8s %6s %9s\n", "method", "futureKL", "cliffJSD", "finalLoss", "maxLoss", "steps", "wallMs")
	b.WriteString("  " + repeat("-", 72) + "\n")
	for _, m := range metrics {
		if failed(m.Method) {
			fmt.Fprintf(b, "  %-14s %9s %10s %10s %8s %6s %9s\n", m.Method, "ERROR", "ERROR", "ERROR", "ERROR", "ERR", "ERR")
			continue
		}
		fk := "-"
		if m.FRPONonZeroLabel() {
			fk = boolStr(m.FutureKLNonZero)
		}
		fmt.Fprintf(b, "  %-14s %9s %10s %10s %8s %6d %9.0f\n",
			m.Method, fk, f(m.CliffJSD), fp(m.FinalLoss), f(m.MaxAbsLoss), m.Steps, m.WallMillis)
	}
}

// sortedKeys returns a map's keys in sorted order for deterministic output.
func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// FRPONonZeroLabel reports whether the futureKL column is meaningful for this row
// (only the FRPO and all-on methods enable the future-KL term).
func (m Metrics) FRPONonZeroLabel() bool {
	return m.FutureKLNonZero || strings.Contains(m.Method, "FRPO") || m.Method == "all-on"
}

func boolStr(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// f formats a float for the table, trimming to a compact fixed form.
func f(v float64) string {
	return fmt.Sprintf("%.4g", v)
}

// fp formats a *float64 (nil -> "null").
func fp(v *float64) string {
	if v == nil {
		return "null"
	}
	return fmt.Sprintf("%.4g", *v)
}

func repeat(s string, n int) string {
	return strings.Repeat(s, n)
}
