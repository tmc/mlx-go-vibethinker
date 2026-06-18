//go:build modelir

package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/tmc/mlx-go-vibethinker/eval/realmodel"
)

// run dispatches to the child (single-method) path when -one-method is set,
// otherwise the parent orchestrator that spawns one child per (method × source).
func run(o opts) error {
	if o.oneMethod >= 0 {
		return runChild(o)
	}
	return runParent(o)
}

func config(o opts) realmodel.Config {
	return realmodel.Config{
		Prompts:     o.prompts,
		K:           o.k,
		MaxTokens:   o.maxTokens,
		Temperature: 0.8,
		Steps:       o.steps,
		LR:          1e-6,
		Seed:        o.seed,
	}
}

func modelDir(o opts) string {
	if o.model != "" {
		return o.model
	}
	return realmodel.DefaultModelDir()
}

// runChild runs exactly one method under one source and prints a single JSON
// ChildRow to stdout, then returns. The ~13GB value-and-grad graph dies when
// this process exits. A failure is still reported as a row (status "error") so
// the parent can place an explicit ERROR cell — the child never exits silently
// without emitting its row.
func runChild(o opts) error {
	src, err := realmodel.ParseSource(o.source)
	if err != nil {
		return err
	}
	row := realmodel.ChildRow{
		Status: "ok",
		Index:  o.oneMethod,
		Source: src.String(),
		Seed:   o.seed,
	}
	mt, err := realmodel.EvaluateOneMethod(context.Background(), modelDir(o), o.oneMethod, config(o), src)
	if err != nil {
		row.Status = "error"
		row.Error = err.Error()
		row.Metrics = realmodel.ErrorMetrics(realmodel.MethodName(o.oneMethod), src.String())
	} else {
		row.Metrics = mt
	}
	out, err := realmodel.MarshalRow(row)
	if err != nil {
		return err
	}
	fmt.Println(string(out))
	return nil
}

// runParent spawns one child process per (method × source), collects the rows,
// turns any crash / timeout / malformed row into an explicit ERROR cell (never a
// silent omission), and emits the combined organic + seeded table or JSON.
func runParent(o opts) error {
	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate self: %w", err)
	}
	dir := modelDir(o)
	cfg := config(o)
	n := realmodel.MethodCount()

	type cell struct {
		metrics realmodel.Metrics
		errMsg  string // non-empty -> ERROR cell
	}
	collect := func(source string) []cell {
		cells := make([]cell, n)
		for idx := 0; idx < n; idx++ {
			fmt.Fprintf(os.Stderr, "[parent] dispatch method %d/%d (%s) source=%s seed=%d tok=%d K=%d steps=%d\n",
				idx, n, realmodel.MethodName(idx), source, o.seed, o.maxTokens, o.k, o.steps)
			m, errMsg := runOneChild(self, o, dir, idx, source)
			if errMsg != "" {
				fmt.Fprintf(os.Stderr, "[parent] method %d (%s/%s) ERROR: %s\n", idx, realmodel.MethodName(idx), source, errMsg)
			}
			cells[idx] = cell{metrics: m, errMsg: errMsg}
		}
		return cells
	}

	organicCells := collect("organic")
	seededCells := collect("seeded")

	organic := make([]realmodel.Metrics, n)
	seeded := make([]realmodel.Metrics, n)
	errs := map[string]string{}
	for i, c := range organicCells {
		organic[i] = c.metrics
		if c.errMsg != "" {
			errs[fmt.Sprintf("ORGANIC/%s", realmodel.MethodName(i))] = c.errMsg
		}
	}
	for i, c := range seededCells {
		seeded[i] = c.metrics
		if c.errMsg != "" {
			errs[fmt.Sprintf("SEEDED/%s", realmodel.MethodName(i))] = c.errMsg
		}
	}

	if o.asJSON {
		doc, err := realmodel.NewReportWithErrors(dir, cfg, organic, seeded, errs).JSON()
		if err != nil {
			return err
		}
		_, err = os.Stdout.Write(doc)
		return err
	}
	fmt.Print(realmodel.TableWithErrors(dir, cfg, organic, seeded, errs))
	return nil
}

// runOneChild executes one child process and parses its row. Any failure mode —
// nonzero exit, no row, malformed/partial JSON, schema mismatch, or a row whose
// own status is "error" — returns an ERROR cell with the child exit code and the
// last stderr line, plus a placeholder Metrics carrying the method/source
// identity so the cell lands in the right place. It never returns a silent gap.
func runOneChild(self string, o opts, dir string, idx int, source string) (realmodel.Metrics, string) {
	name := realmodel.MethodName(idx)
	args := []string{
		"-one-method", strconv.Itoa(idx),
		"-source", source,
		"-model", dir,
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
	placeholder := realmodel.ErrorMetrics(name, srcUpper)

	if runErr != nil {
		return placeholder, fmt.Sprintf("child exited: %v; stderr: %s", runErr, lastLine(stderr.String()))
	}
	out := bytes.TrimSpace(stdout.Bytes())
	if len(out) == 0 {
		return placeholder, fmt.Sprintf("child produced no row; stderr: %s", lastLine(stderr.String()))
	}
	row, err := realmodel.ParseRow(out)
	if err != nil {
		return placeholder, fmt.Sprintf("unparseable child row: %v; stderr: %s", err, lastLine(stderr.String()))
	}
	if row.Index != idx || row.Source != srcUpper {
		return placeholder, fmt.Sprintf("child row identity mismatch: got idx=%d src=%s want idx=%d src=%s", row.Index, row.Source, idx, srcUpper)
	}
	if row.Status == "error" {
		return row.Metrics, fmt.Sprintf("child reported error: %s", row.Error)
	}
	return row.Metrics, ""
}

func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return "(none)"
	}
	lines := strings.Split(s, "\n")
	return lines[len(lines)-1]
}
