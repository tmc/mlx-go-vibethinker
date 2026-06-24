//go:build modelir

package realmodel

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// SweepRowSchema is the version of the two-process sweep child-row contract. The
// parent rejects a row whose schema does not match, so a parent/child binary
// mismatch is caught loudly rather than silently misparsed. It is independent of
// the method-harness RowSchema (different envelope).
const SweepRowSchema = 1

// SweepRow is the fixed-schema envelope ONE sweep child process emits for ONE
// phase of ONE config. The two-process split runs each config as P1 then P2:
//
//	Phase 1 (score@0 + train + checkpoint): emits Acc0, the mechanism stats, the
//	  steps actually taken, and CkptPath — the temp safetensors of the trained q/v
//	  the parent threads into P2.
//	Phase 2 (apply checkpoint + score@final): emits AccFinal.
//
// Exactly one child prints exactly one SweepRow as a single JSON line on stdout,
// then exits — the ~13GB value-and-grad graph (P1) and the non-reclaimable decode
// arrays (both) die with the child. Every field is always present so the parent
// merges by field name. Status is "ok" or "error"; on error the identity fields
// are still set so the parent can place the ERROR cell.
type SweepRow struct {
	Schema     int     `json:"schema"`
	Status     string  `json:"status"` // "ok" | "error"
	Phase      int     `json:"phase"`  // 1 | 2
	Index      int     `json:"index"`  // swept config index (C1..C5)
	Config     string  `json:"config"` // config name
	Source     string  `json:"source"` // "ORGANIC" | "SEEDED"
	Seed       uint64  `json:"seed"`
	Error      string  `json:"error,omitempty"`
	Acc        float64 `json:"acc"`                 // P1: Acc0; P2: AccFinal
	HeldoutN   int     `json:"heldout_n"`           // |held-out probe set|
	StepsRun   int     `json:"steps_run,omitempty"` // P1 only
	CkptPath   string  `json:"ckpt_path,omitempty"` // P1 only: trained-q/v checkpoint
	Mechanism  Metrics `json:"mechanism"`           // P1 only (zero-valued in P2)
	WallMillis float64 `json:"wall_millis"`
}

// MarshalSweepRow renders a sweep child row as a single compact JSON line.
func MarshalSweepRow(r SweepRow) ([]byte, error) {
	r.Schema = SweepRowSchema
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, fmt.Errorf("realmodel: marshal sweep row: %w", err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// ParseSweepRow parses a sweep child row, rejecting a schema mismatch or a
// malformed/partial row loudly so the parent records an ERROR cell rather than
// misreading it.
func ParseSweepRow(data []byte) (SweepRow, error) {
	var r SweepRow
	if err := json.Unmarshal(bytes.TrimSpace(data), &r); err != nil {
		return SweepRow{}, fmt.Errorf("realmodel: parse sweep row: %w", err)
	}
	if r.Schema != SweepRowSchema {
		return SweepRow{}, fmt.Errorf("realmodel: sweep row schema %d != parent schema %d", r.Schema, SweepRowSchema)
	}
	if r.Status != "ok" && r.Status != "error" {
		return SweepRow{}, fmt.Errorf("realmodel: sweep row has invalid status %q", r.Status)
	}
	if r.Phase != 1 && r.Phase != 2 {
		return SweepRow{}, fmt.Errorf("realmodel: sweep row has invalid phase %d", r.Phase)
	}
	return r, nil
}

// RunSweepPhase1 runs PHASE 1 of one config in its own process: load base, score
// the fixed held-out probe at step 0 (Acc0), run the real GRPO training loop, and
// save ONLY the trained q/v projections to ckptPath. It returns the row the
// parent threads into Phase 2 (Acc0, mechanism stats, the checkpoint path). It
// does NOT score the final policy — that is Phase 2, a separate process, so this
// child never piles the final O(n²) decode on top of the training residue (the
// crash the split exists to avoid).
func RunSweepPhase1(ctx context.Context, dir string, idx int, cfg Config, src Source, ckptPath string) (SweepRow, error) {
	cfgs := SweepConfigs()
	if idx < 0 || idx >= len(cfgs) {
		return SweepRow{}, fmt.Errorf("realmodel: swept config index %d out of range [0,%d)", idx, len(cfgs))
	}
	sc := cfgs[idx]
	start := time.Now()
	row := SweepRow{Status: "ok", Phase: 1, Index: idx, Config: sc.Name, Source: src.String(), Seed: cfg.Seed}

	m, err := Load(ctx, dir)
	if err != nil {
		return row, fmt.Errorf("realmodel: load for %q: %w", sc.Name, err)
	}
	defer func() {
		m.Close()
		reclaim()
	}()

	set := HeldoutSet()
	step0, err := scoreHeldout(ctx, m, set, heldoutMaxTokens)
	if err != nil {
		return row, fmt.Errorf("realmodel: held-out step0 for %q: %w", sc.Name, err)
	}
	row.Acc = step0.Acc
	row.HeldoutN = step0.N
	reclaim()

	groups, err := buildGroups(ctx, m, cfg, src)
	if err != nil {
		return row, fmt.Errorf("realmodel: build %s groups for %q: %w", src, sc.Name, err)
	}
	defer freeGroups(groups)
	mech, err := runMethod(ctx, m, sc.Method, cfg, groups)
	if err != nil {
		return row, fmt.Errorf("realmodel: train %q (%s): %w", sc.Name, src, err)
	}
	mech.Source = src.String()
	row.Mechanism = mech
	row.StepsRun = mech.Steps

	n, err := saveTrainedQV(m, ckptPath)
	if err != nil {
		return row, fmt.Errorf("realmodel: save q/v checkpoint for %q: %w", sc.Name, err)
	}
	if n == 0 {
		return row, fmt.Errorf("realmodel: empty q/v checkpoint for %q", sc.Name)
	}
	row.CkptPath = ckptPath
	row.WallMillis = float64(time.Since(start).Milliseconds())
	return row, nil
}

// RunSweepPhase2 runs PHASE 2 of one config in its own process: load base, apply
// the Phase-1 trained-q/v checkpoint, and score the SAME fixed held-out probe at
// the final (trained) policy (AccFinal). A fresh process means the held-out decode
// runs on a clean Metal budget — the same green budget the step-0 pass had — with
// no training residue resident, which is the whole point of the split.
func RunSweepPhase2(ctx context.Context, dir string, idx int, cfg Config, src Source, ckptPath string) (SweepRow, error) {
	cfgs := SweepConfigs()
	if idx < 0 || idx >= len(cfgs) {
		return SweepRow{}, fmt.Errorf("realmodel: swept config index %d out of range [0,%d)", idx, len(cfgs))
	}
	sc := cfgs[idx]
	start := time.Now()
	row := SweepRow{Status: "ok", Phase: 2, Index: idx, Config: sc.Name, Source: src.String(), Seed: cfg.Seed}

	m, err := Load(ctx, dir)
	if err != nil {
		return row, fmt.Errorf("realmodel: load for %q: %w", sc.Name, err)
	}
	defer func() {
		m.Close()
		reclaim()
	}()

	if _, err := loadTrainedQV(m, ckptPath); err != nil {
		return row, fmt.Errorf("realmodel: apply q/v checkpoint for %q: %w", sc.Name, err)
	}
	reclaim()

	set := HeldoutSet()
	final, err := scoreHeldout(ctx, m, set, heldoutMaxTokens)
	if err != nil {
		return row, fmt.Errorf("realmodel: held-out final for %q: %w", sc.Name, err)
	}
	row.Acc = final.Acc
	row.HeldoutN = final.N
	row.WallMillis = float64(time.Since(start).Milliseconds())
	return row, nil
}
