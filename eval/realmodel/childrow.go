//go:build modelir

package realmodel

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// RowSchema is the version of the child-row JSON contract. The parent rejects a
// row whose schema does not match, so a parent/child binary mismatch is caught
// loudly rather than silently misparsed.
const RowSchema = 1

// ChildRow is the fixed-schema envelope one child process emits for one
// (method × source) trial. Exactly one child prints exactly one ChildRow as a
// single JSON object on stdout, then exits. Every field is always present
// (Metrics is itself schema-stable), so the parent merges by field name.
//
// Status is "ok" on success or "error" on failure; on error Metrics still
// carries the method/source identity (so the parent can place the ERROR cell)
// and Error holds the reason.
type ChildRow struct {
	Schema  int     `json:"schema"`
	Status  string  `json:"status"` // "ok" | "error"
	Index   int     `json:"index"`  // registry method index
	Source  string  `json:"source"` // "ORGANIC" | "SEEDED"
	Seed    uint64  `json:"seed"`
	Error   string  `json:"error,omitempty"`
	Metrics Metrics `json:"metrics"`
}

// MarshalRow renders a child row as a single compact JSON line.
func MarshalRow(r ChildRow) ([]byte, error) {
	r.Schema = RowSchema
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		return nil, fmt.Errorf("realmodel: marshal child row: %w", err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// ParseRow parses a child row from stdout bytes, rejecting a schema mismatch or
// a malformed/partial row loudly (so the parent records an ERROR cell rather
// than misreading it).
func ParseRow(data []byte) (ChildRow, error) {
	var r ChildRow
	if err := json.Unmarshal(bytes.TrimSpace(data), &r); err != nil {
		return ChildRow{}, fmt.Errorf("realmodel: parse child row: %w", err)
	}
	if r.Schema != RowSchema {
		return ChildRow{}, fmt.Errorf("realmodel: child row schema %d != parent schema %d", r.Schema, RowSchema)
	}
	if r.Status != "ok" && r.Status != "error" {
		return ChildRow{}, fmt.Errorf("realmodel: child row has invalid status %q", r.Status)
	}
	return r, nil
}

// ErrorMetrics returns a Metrics value that marks a failed trial: the method and
// source identity are filled so the parent can place the cell, and LossFinite is
// false (the trial did not produce a finite-loss run). All numeric fields are
// zero and FinalLoss is nil (JSON null), which the table renders as an ERROR.
func ErrorMetrics(method, source string) Metrics {
	return Metrics{Method: method, Source: source, LossFinite: false}
}
