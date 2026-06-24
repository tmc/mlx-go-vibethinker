//go:build modelir

package realmodel

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// StitchedResult is one config's full sweep outcome, assembled in the parent from
// its Phase-1 row (Acc0 + mechanism stats) and Phase-2 row (AccFinal). DeltaAcc =
// AccFinal − AccStep0 is the directional ranking signal. ErrMsg is non-empty when
// either phase failed; the config is then an explicit ERROR cell, never a silent
// gap, and the Acc fields are not meaningful.
type StitchedResult struct {
	Config     string  `json:"config"`
	Source     string  `json:"source"`
	Seed       uint64  `json:"seed"`
	AccStep0   float64 `json:"acc_step0"`
	AccFinal   float64 `json:"acc_final"`
	DeltaAcc   float64 `json:"delta_acc"`
	HeldoutN   int     `json:"heldout_n"`
	StepsRun   int     `json:"steps_run"`
	Mechanism  Metrics `json:"mechanism"`
	WallMillis float64 `json:"wall_millis"`
	ErrMsg     string  `json:"error,omitempty"`
}

// Stitch assembles a StitchedResult from a config's P1 and P2 rows. If either
// phase errored (ErrMsg passed in by the parent for a crashed/partial child) the
// result is an ERROR cell carrying the identity and the reason. Otherwise the
// held-out delta and the mechanism stats are merged by field name.
func Stitch(p1, p2 SweepRow, p1Err, p2Err string) StitchedResult {
	r := StitchedResult{
		Config:   p1.Config,
		Source:   p1.Source,
		Seed:     p1.Seed,
		HeldoutN: p1.HeldoutN,
	}
	if r.Config == "" { // P1 row missing entirely; fall back to P2 identity
		r.Config, r.Source, r.Seed, r.HeldoutN = p2.Config, p2.Source, p2.Seed, p2.HeldoutN
	}
	if p1Err != "" || p2Err != "" {
		r.ErrMsg = strings.TrimSpace(p1Err + " " + p2Err)
		r.Mechanism = ErrorMetrics(r.Config, r.Source)
		return r
	}
	r.AccStep0 = p1.Acc
	r.AccFinal = p2.Acc
	r.DeltaAcc = p2.Acc - p1.Acc
	r.StepsRun = p1.StepsRun
	r.Mechanism = p1.Mechanism
	r.WallMillis = p1.WallMillis + p2.WallMillis
	return r
}

// SweepReport is the machine-readable differential-training-sweep document: the
// honesty header, the run config, the per-config stitched results, and the
// ranking by held-out Δacc. NoiseFloor (when set, from >=2 baseline seeds) is the
// run-to-run Δacc spread any method must beat to count as real. Recommendation is
// the chosen default config (or the honest "indistinguishable" verdict).
type SweepReport struct {
	Schema          int              `json:"schema"`
	Header          string           `json:"header"`
	Model           string           `json:"model"`
	Harness         string           `json:"harness"`
	NonReproducible bool             `json:"non_reproducible"`
	Note            string           `json:"note"`
	Config          Config           `json:"config"`
	Results         []StitchedResult `json:"results"` // registry/C-order
	Ranking         []string         `json:"ranking"` // config names, best Δacc first
	NoiseFloor      *NoiseFloor      `json:"noise_floor,omitempty"`
	Recommendation  string           `json:"recommendation"`
}

// NoiseFloor is the baseline (C1) run-to-run Δacc spread over >=2 seeds — the bar
// any method's Δacc must clear to count as a real effect rather than optimizer/
// rollout wobble. Seeds and Deltas are aligned.
type NoiseFloor struct {
	Config string    `json:"config"`
	Seeds  []uint64  `json:"seeds"`
	Deltas []float64 `json:"deltas"`
	Min    float64   `json:"min"`
	Max    float64   `json:"max"`
	Spread float64   `json:"spread"` // Max − Min
}

// sweepHeader is the directive's honesty banner: every number is a DIRECTIONAL
// signal on a single M4 Max, not benchmark accuracy.
func sweepHeader(n int) string {
	return fmt.Sprintf("REAL-MODEL DIRECTIONAL TRAINING SWEEP (Qwen2.5-Math-1.5B); held-out N=%d, "+
		"Avg@1 greedy; bounded by Metal array ceiling + subprocess VG-isolation (two-process split: "+
		"score@0+train+checkpoint | apply+score@final); DIRECTIONAL not benchmark accuracy; full repro "+
		"~3.9K H800-hrs is OUT of scope.", n)
}

const sweepNote = "Δacc = AccFinal − AccStep0 over a FIXED held-out probe scored by greedy Avg@1 — a " +
	"DIRECTIONAL signal on unseen short-horizon math, NOT benchmark accuracy. The held-out decode is " +
	"greedy/deterministic, so AccStep0 is identical across seeds (constant base weights); the run-to-run " +
	"variation is in AccFinal, driven by the temperature-sampled training rollouts + the optimizer. Each " +
	"config runs as TWO subprocesses (P1 score@0+train+save-q/v, P2 apply-q/v+score@final) so the ~13GB " +
	"value-and-grad graph and the non-reclaimable held-out decode arrays never share one process's Metal " +
	"budget; a crashed/partial phase is an explicit ERROR cell, never a silent gap. Mechanism stats ride " +
	"alongside as corroborating 'why', NOT as the ranking key. Every number is model- and machine-dependent " +
	"and NOT bit-reproducible across runs."

// RankResults returns the config names ordered by held-out Δacc, best first;
// ERROR cells sort last (they have no meaningful delta). Ties break by config
// name for stable output.
func RankResults(results []StitchedResult) []string {
	idx := make([]int, len(results))
	for i := range results {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ra, rb := results[idx[a]], results[idx[b]]
		ea, eb := ra.ErrMsg != "", rb.ErrMsg != ""
		if ea != eb {
			return !ea // non-error before error
		}
		if ea && eb {
			return ra.Config < rb.Config
		}
		if ra.DeltaAcc != rb.DeltaAcc {
			return ra.DeltaAcc > rb.DeltaAcc
		}
		return ra.Config < rb.Config
	})
	names := make([]string, len(results))
	for i, j := range idx {
		names[i] = results[j].Config
	}
	return names
}

// recommend returns the default-config recommendation. When the best non-error
// Δacc does not clear the noise floor (or no noise floor is known and all deltas
// are ~0), the honest verdict is "indistinguishable" and the recommendation
// defaults to the simplest config (C1 baseline). Otherwise it names the top config.
func recommend(results []StitchedResult, ranking []string, nf *NoiseFloor) string {
	if len(ranking) == 0 {
		return "no configs ran — cannot recommend"
	}
	byName := map[string]StitchedResult{}
	for _, r := range results {
		byName[r.Config] = r
	}
	top := byName[ranking[0]]
	if top.ErrMsg != "" {
		return "all configs errored — cannot recommend (see ERROR cells)"
	}
	floor := 0.0
	floorSrc := "0 (no >=2-seed noise floor measured)"
	if nf != nil {
		floor = nf.Spread
		floorSrc = fmt.Sprintf("the %s noise floor spread %.3f over seeds %v", nf.Config, nf.Spread, nf.Seeds)
	}
	if top.DeltaAcc <= floor {
		return fmt.Sprintf("INDISTINGUISHABLE at N=%d: best Δacc (%s, %+.3f) does not clear %s; "+
			"default to the simplest config (C1-baseline). A bigger held-out set / more steps would be "+
			"needed to separate the methods — out of scope on a single M4 Max.",
			top.HeldoutN, top.Config, top.DeltaAcc, floorSrc)
	}
	return fmt.Sprintf("RECOMMEND %s: Δacc %+.3f clears %s. Mechanism stats corroborate (see table).",
		top.Config, top.DeltaAcc, floorSrc)
}

// NewSweepReport assembles the full report from the run config, the stitched
// per-config results, and (optionally) the baseline noise floor.
func NewSweepReport(model string, cfg Config, results []StitchedResult, nf *NoiseFloor) SweepReport {
	ranking := RankResults(results)
	return SweepReport{
		Schema:          SweepRowSchema,
		Header:          sweepHeader(heldoutN(results)),
		Model:           model,
		Harness:         "two-process-per-config split (score@0+train+checkpoint | apply+score@final); VG + decode isolation",
		NonReproducible: true,
		Note:            sweepNote,
		Config:          cfg,
		Results:         results,
		Ranking:         ranking,
		NoiseFloor:      nf,
		Recommendation:  recommend(results, ranking, nf),
	}
}

func heldoutN(results []StitchedResult) int {
	for _, r := range results {
		if r.HeldoutN > 0 {
			return r.HeldoutN
		}
	}
	return 0
}

// JSON renders the sweep report as deterministic indented JSON.
func (r SweepReport) JSON() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, fmt.Errorf("realmodel: encode sweep report: %w", err)
	}
	return buf.Bytes(), nil
}

// SweepTable renders the human-readable sweep: the honesty header, the ranking by
// held-out Δacc with mechanism stats alongside, the noise floor, and the
// recommendation. ERROR cells are rendered explicitly.
func (r SweepReport) SweepTable() string {
	var b strings.Builder
	b.WriteString(r.Header)
	b.WriteString("\n\n")
	fmt.Fprintf(&b, "model: %s\nconfig: prompts=%d K=%d maxTok=%d temp=%.2f steps=%d lr=%.0e seed=%d source=%s\n\n",
		r.Model, r.Config.Prompts, r.Config.K, r.Config.MaxTokens, r.Config.Temperature, r.Config.Steps, r.Config.LR, r.Config.Seed,
		firstSource(r.Results))

	byName := map[string]StitchedResult{}
	for _, res := range r.Results {
		byName[res.Config] = res
	}

	b.WriteString("RANKING by held-out Δacc (final − step0); mechanism stats alongside as corroborating 'why'\n")
	fmt.Fprintf(&b, "  %-14s %8s %8s %8s %6s %10s %10s %10s\n",
		"config", "acc0", "accFin", "Δacc", "steps", "ratioVar", "advStd", "wallS")
	b.WriteString("  " + strings.Repeat("-", 84) + "\n")
	for rank, name := range r.Ranking {
		res := byName[name]
		if res.ErrMsg != "" {
			fmt.Fprintf(&b, "  %-14s %8s %8s %8s %6s %10s %10s %10s   <- ERROR\n",
				fmt.Sprintf("%d.%s", rank+1, name), "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR")
			continue
		}
		fmt.Fprintf(&b, "  %-14s %8.3f %8.3f %+8.3f %6d %10s %10s %10.1f\n",
			fmt.Sprintf("%d.%s", rank+1, name), res.AccStep0, res.AccFinal, res.DeltaAcc, res.StepsRun,
			f(res.Mechanism.RatioVar), f(res.Mechanism.AdvStd), res.WallMillis/1000)
	}
	b.WriteString("\n")

	if r.NoiseFloor != nil {
		nf := r.NoiseFloor
		fmt.Fprintf(&b, "NOISE FLOOR (%s, %d seeds %v): Δacc deltas %v -> spread %.3f (min %+.3f, max %+.3f)\n",
			nf.Config, len(nf.Seeds), nf.Seeds, nf.Deltas, nf.Spread, nf.Min, nf.Max)
		b.WriteString("  Any method's Δacc must clear this spread to count as a real effect, not run-to-run wobble.\n\n")
	} else {
		b.WriteString("NOISE FLOOR: not measured (single seed) — Δacc differences below run-to-run wobble cannot be trusted.\n\n")
	}

	fmt.Fprintf(&b, "RECOMMENDATION: %s\n", r.Recommendation)

	if errs := errorCells(r.Results); len(errs) > 0 {
		b.WriteString("\n!!! ERROR CELLS !!!\n")
		for _, name := range errs {
			fmt.Fprintf(&b, "  %s: %s\n", name, byName[name].ErrMsg)
		}
	}
	b.WriteString("\nNOTE: " + sweepNote + "\n")
	return b.String()
}

func firstSource(results []StitchedResult) string {
	for _, r := range results {
		if r.Source != "" {
			return r.Source
		}
	}
	return "?"
}

func errorCells(results []StitchedResult) []string {
	var out []string
	for _, r := range results {
		if r.ErrMsg != "" {
			out = append(out, r.Config)
		}
	}
	sort.Strings(out)
	return out
}
