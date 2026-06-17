package ssp

import (
	"context"
	"fmt"
	"maps"
)

// A Checkpoint identifies a model state produced by a stage. It is the typed
// artifact threaded between stages: a stage loads weights from Dir, trains or
// transforms them, writes new weights to a fresh directory, and returns a new
// Checkpoint pointing at it.
//
// The zero Checkpoint is usable as a pipeline's initial input only if a stage
// is prepared to start from an empty Dir (for example a stage that loads a base
// model by name from Params); otherwise supply a Checkpoint whose Dir holds the
// starting weights.
type Checkpoint struct {
	// Dir is the directory holding this checkpoint's model weights and
	// config. An empty Dir denotes "no materialized weights yet" — the
	// starting point of a pipeline whose first stage loads a base model.
	Dir string

	// Params carries free-form scalar metadata about the checkpoint (for
	// example the base model name or the recipe label). It is copied, not
	// shared, when a stage derives a new checkpoint with deriveCheckpoint.
	Params map[string]string

	// History records, oldest first, the provenance of every stage that has
	// contributed to this checkpoint. The last entry describes the stage
	// that produced this checkpoint.
	History []Provenance
}

// Provenance records that a stage produced a checkpoint from a given input.
type Provenance struct {
	// Stage is the name of the stage that produced the checkpoint.
	Stage string

	// InputDir is the Dir of the checkpoint the stage consumed. It is empty
	// when the stage started from the zero checkpoint.
	InputDir string

	// OutputDir is the Dir of the checkpoint the stage produced.
	OutputDir string

	// Note is an optional human-readable summary of what the stage did.
	Note string
}

// A Stage transforms one checkpoint into another. Implementations live in the
// other packages of this module (spectrum, signal, distill, instruct); ssp only
// orchestrates them.
//
// Run must not mutate its input checkpoint. It returns a new checkpoint, or an
// error; on error the pipeline stops and the partial result is discarded. Run
// should honor ctx cancellation between expensive sub-steps.
type Stage interface {
	// Name returns a short, stable identifier for the stage, used in
	// provenance and progress reporting.
	Name() string

	// Run transforms in into a new checkpoint. in is never nil; callers that
	// have no starting weights pass a checkpoint with an empty Dir.
	Run(ctx context.Context, in *Checkpoint) (*Checkpoint, error)
}

// StageFunc adapts a plain function to the Stage interface. The returned stage
// reports name from Name.
type StageFunc struct {
	StageName string
	Fn        func(ctx context.Context, in *Checkpoint) (*Checkpoint, error)
}

// Name reports the stage name.
func (s StageFunc) Name() string { return s.StageName }

// Run calls the wrapped function.
func (s StageFunc) Run(ctx context.Context, in *Checkpoint) (*Checkpoint, error) {
	if s.Fn == nil {
		return nil, fmt.Errorf("ssp: StageFunc %q has nil Fn", s.StageName)
	}
	return s.Fn(ctx, in)
}

// A Pipeline is an ordered list of stages forming one training recipe. The
// zero Pipeline runs no stages and returns its input unchanged.
type Pipeline struct {
	// Name labels the recipe (for example "vibethinker-1.5b").
	Name string

	// Stages are run in order; each consumes the previous stage's output.
	Stages []Stage

	// Observe, if non-nil, is called before and after each stage with the
	// stage name and, on completion, the resulting checkpoint or error. It is
	// a reporting hook only and must not modify the checkpoint.
	Observe func(ev Event)
}

// EventKind classifies a pipeline Event.
type EventKind int

const (
	// StageStart is emitted just before a stage runs.
	StageStart EventKind = iota
	// StageDone is emitted after a stage returns successfully.
	StageDone
	// StageError is emitted after a stage returns an error.
	StageError
)

// An Event reports pipeline progress to a Pipeline.Observe hook.
type Event struct {
	Kind  EventKind
	Index int    // zero-based index of the stage in Stages
	Stage string // stage name
	Out   *Checkpoint
	Err   error
}

// Run executes the pipeline's stages in order, threading the checkpoint and
// recording provenance. It returns the final checkpoint, or the first stage
// error encountered. A nil initial checkpoint is treated as the zero
// checkpoint. The input checkpoint is never mutated.
func (p *Pipeline) Run(ctx context.Context, initial *Checkpoint) (*Checkpoint, error) {
	cur := cloneCheckpoint(initial)
	for i, st := range p.Stages {
		if st == nil {
			return nil, fmt.Errorf("ssp: pipeline %q stage %d is nil", p.Name, i)
		}
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("ssp: pipeline %q canceled before stage %q: %w", p.Name, st.Name(), err)
		}
		p.emit(Event{Kind: StageStart, Index: i, Stage: st.Name()})
		out, err := st.Run(ctx, cur)
		if err != nil {
			p.emit(Event{Kind: StageError, Index: i, Stage: st.Name(), Err: err})
			return nil, fmt.Errorf("ssp: pipeline %q stage %d %q: %w", p.Name, i, st.Name(), err)
		}
		if out == nil {
			err := fmt.Errorf("ssp: pipeline %q stage %d %q returned nil checkpoint", p.Name, i, st.Name())
			p.emit(Event{Kind: StageError, Index: i, Stage: st.Name(), Err: err})
			return nil, err
		}
		p.emit(Event{Kind: StageDone, Index: i, Stage: st.Name(), Out: out})
		cur = out
	}
	return cur, nil
}

func (p *Pipeline) emit(ev Event) {
	if p.Observe != nil {
		p.Observe(ev)
	}
}

// Derive builds the checkpoint a stage should return: it copies in's params and
// history, appends a Provenance entry for the named stage, and sets the output
// directory. Stage implementations call Derive instead of constructing a
// Checkpoint by hand so provenance is recorded uniformly.
//
// in may be nil (treated as the zero checkpoint). outDir is the directory the
// stage wrote its new weights to. note is an optional summary.
func Derive(in *Checkpoint, stage, outDir, note string) *Checkpoint {
	out := cloneCheckpoint(in)
	inDir := ""
	if in != nil {
		inDir = in.Dir
	}
	out.Dir = outDir
	out.History = append(out.History, Provenance{
		Stage:     stage,
		InputDir:  inDir,
		OutputDir: outDir,
		Note:      note,
	})
	return out
}

// cloneCheckpoint returns a deep-enough copy of c: Params and History are
// copied so callers cannot mutate a shared input. A nil c yields a fresh zero
// checkpoint.
func cloneCheckpoint(c *Checkpoint) *Checkpoint {
	out := &Checkpoint{}
	if c == nil {
		out.Params = map[string]string{}
		return out
	}
	out.Dir = c.Dir
	out.Params = make(map[string]string, len(c.Params))
	maps.Copy(out.Params, c.Params)
	if len(c.History) > 0 {
		out.History = make([]Provenance, len(c.History))
		copy(out.History, c.History)
	}
	return out
}
