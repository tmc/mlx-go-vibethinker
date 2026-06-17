package ssp

import (
	"context"
	"errors"
	"testing"
)

// recordStage is a test stage that derives a new checkpoint and records the
// inputs it saw.
type recordStage struct {
	name   string
	outDir string
	err    error
	sawDir *string // optional: records in.Dir when run
}

func (s recordStage) Name() string { return s.name }

func (s recordStage) Run(ctx context.Context, in *Checkpoint) (*Checkpoint, error) {
	if s.sawDir != nil {
		*s.sawDir = in.Dir
	}
	if s.err != nil {
		return nil, s.err
	}
	return Derive(in, s.name, s.outDir, ""), nil
}

func TestPipelineThreadsCheckpointAndProvenance(t *testing.T) {
	var saw2 string
	p := &Pipeline{
		Name: "test",
		Stages: []Stage{
			recordStage{name: "a", outDir: "/ck/a"},
			recordStage{name: "b", outDir: "/ck/b", sawDir: &saw2},
		},
	}
	out, err := p.Run(context.Background(), &Checkpoint{Dir: "/ck/base"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Stage b must have seen stage a's output dir as its input.
	if saw2 != "/ck/a" {
		t.Fatalf("stage b saw input %q, want /ck/a", saw2)
	}
	if out.Dir != "/ck/b" {
		t.Fatalf("final dir = %q, want /ck/b", out.Dir)
	}
	// Provenance is oldest-first, one entry per stage.
	wantStages := []string{"a", "b"}
	if len(out.History) != len(wantStages) {
		t.Fatalf("history len = %d, want %d: %+v", len(out.History), len(wantStages), out.History)
	}
	for i, w := range wantStages {
		if out.History[i].Stage != w {
			t.Fatalf("history[%d].Stage = %q, want %q", i, out.History[i].Stage, w)
		}
	}
	// Chain: a consumed base, b consumed a.
	if out.History[0].InputDir != "/ck/base" || out.History[0].OutputDir != "/ck/a" {
		t.Fatalf("history[0] = %+v", out.History[0])
	}
	if out.History[1].InputDir != "/ck/a" || out.History[1].OutputDir != "/ck/b" {
		t.Fatalf("history[1] = %+v", out.History[1])
	}
}

func TestPipelineDoesNotMutateInput(t *testing.T) {
	in := &Checkpoint{Dir: "/ck/base", Params: map[string]string{"k": "v"}}
	p := &Pipeline{Stages: []Stage{recordStage{name: "a", outDir: "/ck/a"}}}
	if _, err := p.Run(context.Background(), in); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if in.Dir != "/ck/base" || len(in.History) != 0 || in.Params["k"] != "v" {
		t.Fatalf("input mutated: %+v", in)
	}
}

func TestPipelineStopsOnError(t *testing.T) {
	sentinel := errors.New("boom")
	var ran3 bool
	p := &Pipeline{
		Name: "test",
		Stages: []Stage{
			recordStage{name: "a", outDir: "/ck/a"},
			recordStage{name: "b", err: sentinel},
			StageFunc{StageName: "c", Fn: func(ctx context.Context, in *Checkpoint) (*Checkpoint, error) {
				ran3 = true
				return Derive(in, "c", "/ck/c", ""), nil
			}},
		},
	}
	_, err := p.Run(context.Background(), &Checkpoint{})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want wrap of sentinel", err)
	}
	if ran3 {
		t.Fatal("stage c ran after stage b failed")
	}
}

func TestPipelineEmptyReturnsInput(t *testing.T) {
	in := &Checkpoint{Dir: "/ck/base"}
	p := &Pipeline{}
	out, err := p.Run(context.Background(), in)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Dir != "/ck/base" {
		t.Fatalf("out.Dir = %q, want /ck/base", out.Dir)
	}
}

func TestPipelineNilInitialIsZeroCheckpoint(t *testing.T) {
	var saw string
	p := &Pipeline{Stages: []Stage{recordStage{name: "a", outDir: "/ck/a", sawDir: &saw}}}
	out, err := p.Run(context.Background(), nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if saw != "" {
		t.Fatalf("first stage saw dir %q, want empty", saw)
	}
	if out.History[0].InputDir != "" {
		t.Fatalf("history[0].InputDir = %q, want empty", out.History[0].InputDir)
	}
}

func TestPipelineRejectsNilStageAndNilOutput(t *testing.T) {
	p := &Pipeline{Stages: []Stage{nil}}
	if _, err := p.Run(context.Background(), &Checkpoint{}); err == nil {
		t.Fatal("want error for nil stage")
	}
	p2 := &Pipeline{Stages: []Stage{StageFunc{StageName: "x", Fn: func(ctx context.Context, in *Checkpoint) (*Checkpoint, error) {
		return nil, nil
	}}}}
	if _, err := p2.Run(context.Background(), &Checkpoint{}); err == nil {
		t.Fatal("want error for nil output checkpoint")
	}
}

func TestPipelineCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &Pipeline{Stages: []Stage{recordStage{name: "a", outDir: "/ck/a"}}}
	if _, err := p.Run(ctx, &Checkpoint{}); err == nil {
		t.Fatal("want error for canceled context")
	}
}

func TestObserveHook(t *testing.T) {
	var kinds []EventKind
	p := &Pipeline{
		Stages:  []Stage{recordStage{name: "a", outDir: "/ck/a"}},
		Observe: func(ev Event) { kinds = append(kinds, ev.Kind) },
	}
	if _, err := p.Run(context.Background(), &Checkpoint{}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(kinds) != 2 || kinds[0] != StageStart || kinds[1] != StageDone {
		t.Fatalf("events = %v, want [StageStart StageDone]", kinds)
	}
}

func TestDeriveCopiesParams(t *testing.T) {
	in := &Checkpoint{Dir: "/a", Params: map[string]string{"base": "qwen"}}
	out := Derive(in, "s", "/b", "note")
	out.Params["base"] = "mutated"
	if in.Params["base"] != "qwen" {
		t.Fatal("Derive shared Params map with input")
	}
	if out.History[len(out.History)-1].Note != "note" {
		t.Fatal("note not recorded")
	}
}
