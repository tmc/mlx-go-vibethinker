package probe

import (
	"context"
	"errors"
	"testing"
)

// fakeEval returns a fixed Pass@K per (checkpoint, subdomain) from a table, so
// argmax selection is exercised without any model. A missing entry scores 0.
type fakeEval struct {
	scores map[string]map[string]float64 // ckpt -> subdomain -> passK
	calls  []string                      // ckpt|sub it was asked, in order
	failOn string                        // checkpoint dir that errors
	err    error
}

func (f *fakeEval) Probe(ctx context.Context, ckpt string, sub Subdomain) (float64, error) {
	f.calls = append(f.calls, ckpt+"|"+sub.Name)
	if ckpt == f.failOn {
		return 0, f.err
	}
	return f.scores[ckpt][sub.Name], nil
}

func mathSubs() []Subdomain {
	subs := make([]Subdomain, len(MathSubdomains))
	for i, n := range MathSubdomains {
		subs[i] = Subdomain{Name: n, NSamples: 8, PassAtK: 4}
	}
	return subs
}

// Property (DESIGN §4.1): Select picks the checkpoint with the highest
// Pass@K per subdomain (argmax over t of Pᵢ(t)) — a different specialist may win
// each subdomain.
func TestSelectArgmaxPerSubdomain(t *testing.T) {
	cks := []Checkpoint{
		{Step: 100, Dir: "/ck/100"},
		{Step: 200, Dir: "/ck/200"},
		{Step: 300, Dir: "/ck/300"},
	}
	eval := &fakeEval{scores: map[string]map[string]float64{
		"/ck/100": {"algebra": 0.9, "geometry": 0.1, "calculus": 0.2, "statistics": 0.3},
		"/ck/200": {"algebra": 0.5, "geometry": 0.8, "calculus": 0.2, "statistics": 0.7},
		"/ck/300": {"algebra": 0.4, "geometry": 0.3, "calculus": 0.95, "statistics": 0.7},
	}}
	got, err := Select(context.Background(), eval, cks, mathSubs())
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	want := map[string]struct {
		dir  string
		step int
		pk   float64
	}{
		"algebra":    {"/ck/100", 100, 0.9},
		"geometry":   {"/ck/200", 200, 0.8},
		"calculus":   {"/ck/300", 300, 0.95},
		"statistics": {"/ck/200", 200, 0.7}, // tie 0.7 with /ck/300; earliest wins
	}
	if len(got) != len(want) {
		t.Fatalf("got %d specialists, want %d", len(got), len(want))
	}
	for _, s := range got {
		w, ok := want[s.Subdomain]
		if !ok {
			t.Fatalf("unexpected subdomain %q", s.Subdomain)
		}
		if s.Checkpoint != w.dir || s.Step != w.step || s.PassK != w.pk {
			t.Errorf("subdomain %q = {%s step=%d pk=%v}, want {%s step=%d pk=%v}",
				s.Subdomain, s.Checkpoint, s.Step, s.PassK, w.dir, w.step, w.pk)
		}
	}
}

// Property: ties are broken deterministically toward the earliest checkpoint in
// the input order — a later checkpoint with an equal score never displaces it.
func TestSelectTieBreakEarliest(t *testing.T) {
	cks := []Checkpoint{
		{Step: 1, Dir: "/a"},
		{Step: 2, Dir: "/b"},
		{Step: 3, Dir: "/c"},
	}
	// All three score identically on the one subdomain.
	eval := &fakeEval{scores: map[string]map[string]float64{
		"/a": {"algebra": 0.5},
		"/b": {"algebra": 0.5},
		"/c": {"algebra": 0.5},
	}}
	subs := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 2}}
	got, err := Select(context.Background(), eval, cks, subs)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got[0].Checkpoint != "/a" || got[0].Step != 1 {
		t.Fatalf("tie not broken to earliest: got %s step=%d", got[0].Checkpoint, got[0].Step)
	}
	// Reordering the input must move the winner to the new earliest.
	rev := []Checkpoint{cks[2], cks[1], cks[0]}
	got2, err := Select(context.Background(), eval, rev, subs)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got2[0].Checkpoint != "/c" {
		t.Fatalf("tie should follow input order: got %s, want /c", got2[0].Checkpoint)
	}
}

// A single checkpoint is selected for every subdomain.
func TestSelectSingleCheckpoint(t *testing.T) {
	cks := []Checkpoint{{Step: 7, Dir: "/only"}}
	eval := &fakeEval{scores: map[string]map[string]float64{
		"/only": {"algebra": 0.3, "geometry": 0.0, "calculus": 0.6, "statistics": 0.6},
	}}
	got, err := Select(context.Background(), eval, cks, mathSubs())
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	for _, s := range got {
		if s.Checkpoint != "/only" || s.Step != 7 {
			t.Fatalf("subdomain %q picked %s step=%d, want /only step=7", s.Subdomain, s.Checkpoint, s.Step)
		}
	}
}

func TestSelectPropagatesEvalError(t *testing.T) {
	sentinel := errors.New("model boom")
	cks := []Checkpoint{{Step: 1, Dir: "/ok"}, {Step: 2, Dir: "/bad"}}
	eval := &fakeEval{
		scores: map[string]map[string]float64{"/ok": {"algebra": 0.5}},
		failOn: "/bad",
		err:    sentinel,
	}
	subs := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 2}}
	if _, err := Select(context.Background(), eval, cks, subs); !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want wrap of sentinel", err)
	}
}

func TestSelectRespectsContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cks := []Checkpoint{{Step: 1, Dir: "/a"}}
	eval := &fakeEval{scores: map[string]map[string]float64{"/a": {"algebra": 1}}}
	subs := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 2}}
	if _, err := Select(ctx, eval, cks, subs); err == nil {
		t.Fatal("want error on canceled context")
	}
}

func TestSelectValidation(t *testing.T) {
	cks := []Checkpoint{{Step: 1, Dir: "/a"}}
	eval := &fakeEval{scores: map[string]map[string]float64{"/a": {"algebra": 1}}}
	good := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 2}}

	if _, err := Select(context.Background(), nil, cks, good); err == nil {
		t.Error("nil evaluator should error")
	}
	if _, err := Select(context.Background(), eval, nil, good); err == nil {
		t.Error("no checkpoints should error")
	}
	if _, err := Select(context.Background(), eval, cks, nil); err == nil {
		t.Error("no subdomains should error")
	}
	// PassAtK out of range.
	bad := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 5}}
	if _, err := Select(context.Background(), eval, cks, bad); err == nil {
		t.Error("PassAtK > NSamples should error")
	}
	bad2 := []Subdomain{{Name: "algebra", NSamples: 4, PassAtK: 0}}
	if _, err := Select(context.Background(), eval, cks, bad2); err == nil {
		t.Error("PassAtK < 1 should error")
	}
	bad3 := []Subdomain{{Name: "algebra", NSamples: 0, PassAtK: 1}}
	if _, err := Select(context.Background(), eval, cks, bad3); err == nil {
		t.Error("NSamples < 1 should error")
	}
}
