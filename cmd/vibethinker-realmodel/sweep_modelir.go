//go:build modelir

package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/mlx-go-vibethinker/eval/realmodel"
)

// runSweepPhaseChild runs ONE phase of ONE config in this process and prints a
// single JSON SweepRow, then returns. Phase 1 = score@0 + train + save-q/v; phase
// 2 = apply-q/v + score@final. The ~13GB graph (P1) and the non-reclaimable decode
// arrays die when this process exits — the isolation the split depends on. A
// failure is still reported as a row (status "error") so the parent can place an
// explicit ERROR cell; the child never exits silently without emitting its row.
func runSweepPhaseChild(o opts) error {
	src, err := realmodel.ParseSource(o.source)
	if err != nil {
		return err
	}
	cfg := config(o)
	dir := modelDir(o)
	ctx := context.Background()

	var row realmodel.SweepRow
	var runErr error
	switch o.sweepPhase {
	case 1:
		row, runErr = realmodel.RunSweepPhase1(ctx, dir, o.sweepConfig, cfg, src, o.ckpt)
	case 2:
		row, runErr = realmodel.RunSweepPhase2(ctx, dir, o.sweepConfig, cfg, src, o.ckpt)
	default:
		return fmt.Errorf("invalid -sweep-phase %d (want 1 or 2)", o.sweepPhase)
	}
	if runErr != nil {
		row.Status = "error"
		row.Error = runErr.Error()
		if row.Config == "" {
			row.Config = realmodel.SweepConfigName(o.sweepConfig)
		}
		if row.Source == "" {
			row.Source = src.String()
		}
		if row.Phase == 0 {
			row.Phase = o.sweepPhase
		}
	}
	out, err := realmodel.MarshalSweepRow(row)
	if err != nil {
		return err
	}
	fmt.Println(string(out))
	return nil
}

// runSweepParent drives the differential training sweep. For each config C1..C5 it
// spawns a Phase-1 child (score@0 + train + checkpoint q/v) then a Phase-2 child
// (apply that checkpoint + score@final), stitches the two rows into a held-out
// Δacc, ranks the configs, and — when -seeds lists >=2 baseline seeds — measures
// the C1 noise floor by running the baseline at each seed. Any child crash / OOM /
// partial row becomes an explicit ERROR cell (exit code + last stderr), never a
// silent gap. The combined table (or -json) carries the honesty header.
func runSweepParent(o opts) error {
	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate self: %w", err)
	}
	dir := modelDir(o)
	cfg := config(o)

	source := o.source
	if source == "" {
		source = "seeded" // the source on which the reward-shape mechanisms are observable
	}

	tmp, err := os.MkdirTemp("", "vibethinker-sweep-")
	if err != nil {
		return fmt.Errorf("sweep tempdir: %w", err)
	}
	defer os.RemoveAll(tmp)

	n := realmodel.SweepConfigCount()
	results := make([]realmodel.StitchedResult, n)
	for idx := 0; idx < n; idx++ {
		name := realmodel.SweepConfigName(idx)
		ckpt := filepath.Join(tmp, fmt.Sprintf("qv-%s-seed%d.safetensors", name, o.seed))
		fmt.Fprintf(os.Stderr, "[sweep] config %d/%d %s seed=%d source=%s tok=%d K=%d steps=%d prompts=%d\n",
			idx, n, name, o.seed, source, o.maxTokens, o.k, o.steps, o.prompts)

		p1, p1Err := runSweepPhase(self, o, dir, idx, source, 1, ckpt)
		if p1Err != "" {
			fmt.Fprintf(os.Stderr, "[sweep] %s P1 ERROR: %s\n", name, p1Err)
			results[idx] = realmodel.Stitch(p1, realmodel.SweepRow{}, p1Err, "phase 2 skipped (phase 1 failed)")
			continue
		}
		p2, p2Err := runSweepPhase(self, o, dir, idx, source, 2, p1.CkptPath)
		if p2Err != "" {
			fmt.Fprintf(os.Stderr, "[sweep] %s P2 ERROR: %s\n", name, p2Err)
		}
		results[idx] = realmodel.Stitch(p1, p2, "", p2Err)
	}

	// Noise floor: rerun the C1 baseline (config index 0) at each extra seed and
	// summarize the Δacc spread. The first listed seed reuses the main run above.
	nf := measureNoiseFloor(self, o, dir, source, tmp, results)

	report := realmodel.NewSweepReport(dir, cfg, results, nf)
	if o.asJSON {
		doc, err := report.JSON()
		if err != nil {
			return err
		}
		_, err = os.Stdout.Write(doc)
		return err
	}
	fmt.Print(report.SweepTable())
	return nil
}

// runSweepPhase spawns one phase child and parses its SweepRow. Every failure mode
// — nonzero exit, no row, malformed/partial JSON, schema mismatch, identity
// mismatch, or a row whose own status is "error" — returns the row (with identity
// when available) plus a non-empty error string carrying the exit cause and the
// last stderr line. It never returns a silent gap.
func runSweepPhase(self string, o opts, dir string, idx int, source string, phase int, ckpt string) (realmodel.SweepRow, string) {
	args := []string{
		"-sweep-phase", strconv.Itoa(phase),
		"-sweep-config", strconv.Itoa(idx),
		"-source", source,
		"-model", dir,
		"-ckpt", ckpt,
		"-prompts", strconv.Itoa(o.prompts),
		"-k", strconv.Itoa(o.k),
		"-max-tokens", strconv.Itoa(o.maxTokens),
		"-steps", strconv.Itoa(o.steps),
		"-seed", strconv.FormatUint(o.seed, 10),
	}
	cmd := exec.Command(self, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	srcUpper := strings.ToUpper(source)
	if runErr != nil {
		return realmodel.SweepRow{}, fmt.Sprintf("phase %d child exited: %v; stderr: %s", phase, runErr, lastLine(stderr.String()))
	}
	out := bytes.TrimSpace(stdout.Bytes())
	if len(out) == 0 {
		return realmodel.SweepRow{}, fmt.Sprintf("phase %d child produced no row; stderr: %s", phase, lastLine(stderr.String()))
	}
	row, err := realmodel.ParseSweepRow(out)
	if err != nil {
		return realmodel.SweepRow{}, fmt.Sprintf("phase %d unparseable row: %v; stderr: %s", phase, err, lastLine(stderr.String()))
	}
	if row.Index != idx || row.Source != srcUpper || row.Phase != phase {
		return row, fmt.Sprintf("phase %d row identity mismatch: got idx=%d src=%s phase=%d want idx=%d src=%s phase=%d",
			phase, row.Index, row.Source, row.Phase, idx, srcUpper, phase)
	}
	if row.Status == "error" {
		return row, fmt.Sprintf("phase %d child reported error: %s", phase, row.Error)
	}
	return row, ""
}

// measureNoiseFloor runs the C1 baseline at each extra seed in -seeds and returns
// the Δacc spread. It returns nil when fewer than two seeds are available (a noise
// floor needs >=2). The first seed reuses the baseline result already in results
// (results[0]) so its child pair is not re-run.
func measureNoiseFloor(self string, o opts, dir, source, tmp string, results []realmodel.StitchedResult) *realmodel.NoiseFloor {
	seeds := parseSeeds(o.seeds, o.seed)
	if len(seeds) < 2 {
		return nil
	}
	deltas := make([]float64, 0, len(seeds))
	used := make([]uint64, 0, len(seeds))
	for _, s := range seeds {
		if s == o.seed && results[0].ErrMsg == "" {
			deltas = append(deltas, results[0].DeltaAcc)
			used = append(used, s)
			continue
		}
		o2 := o
		o2.seed = s
		ckpt := filepath.Join(tmp, fmt.Sprintf("qv-C1-noise-seed%d.safetensors", s))
		fmt.Fprintf(os.Stderr, "[sweep] noise-floor C1 baseline seed=%d\n", s)
		p1, p1Err := runSweepPhase(self, o2, dir, 0, source, 1, ckpt)
		if p1Err != "" {
			fmt.Fprintf(os.Stderr, "[sweep] noise-floor seed=%d P1 ERROR: %s\n", s, p1Err)
			continue
		}
		p2, p2Err := runSweepPhase(self, o2, dir, 0, source, 2, p1.CkptPath)
		if p2Err != "" {
			fmt.Fprintf(os.Stderr, "[sweep] noise-floor seed=%d P2 ERROR: %s\n", s, p2Err)
			continue
		}
		st := realmodel.Stitch(p1, p2, "", "")
		deltas = append(deltas, st.DeltaAcc)
		used = append(used, s)
	}
	if len(deltas) < 2 {
		return nil
	}
	min, max, spread := realmodel.DeltaSpread(deltas)
	return &realmodel.NoiseFloor{
		Config: realmodel.SweepConfigName(0),
		Seeds:  used,
		Deltas: deltas,
		Min:    min,
		Max:    max,
		Spread: spread,
	}
}

// parseSeeds parses the comma-separated -seeds list, always including the primary
// seed, de-duplicated, preserving first-seen order.
func parseSeeds(s string, primary uint64) []uint64 {
	seen := map[uint64]bool{}
	out := []uint64{}
	add := func(v uint64) {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	add(primary)
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			continue
		}
		add(v)
	}
	return out
}
