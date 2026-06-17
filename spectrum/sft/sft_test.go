package sft

import (
	"context"
	"math"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

func TestPackGreedyFirstFit(t *testing.T) {
	// blockSize 5: [3,2] fit in one block; the next [4] starts a new block.
	seqs := [][]int{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}}
	res, err := Pack(seqs, 5, 0)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if len(res.Blocks) != 2 {
		t.Fatalf("got %d blocks, want 2", len(res.Blocks))
	}
	b0 := res.Blocks[0]
	if want := []int{1, 2, 3, 4, 5}; !eqInt(b0.Tokens, want) {
		t.Fatalf("block0 tokens = %v, want %v", b0.Tokens, want)
	}
	if !eqInt(b0.SegmentLengths, []int{3, 2}) {
		t.Fatalf("block0 seglens = %v, want [3 2]", b0.SegmentLengths)
	}
	// segment ids: 0,0,0,1,1
	if !eqInt(b0.SegmentID, []int{0, 0, 0, 1, 1}) {
		t.Fatalf("block0 segIDs = %v", b0.SegmentID)
	}
	// block1: [6,7,8,9] + 1 pad
	b1 := res.Blocks[1]
	if !eqInt(b1.Tokens, []int{6, 7, 8, 9, 0}) {
		t.Fatalf("block1 tokens = %v", b1.Tokens)
	}
	if b1.SegmentID[4] != -1 {
		t.Fatalf("pad position should have segID -1, got %d", b1.SegmentID[4])
	}
	if b1.RealTokens() != 4 {
		t.Fatalf("block1 real tokens = %d, want 4", b1.RealTokens())
	}
	if !eqF32(b1.LossMask(), []float32{1, 1, 1, 1, 0}) {
		t.Fatalf("block1 loss mask = %v", b1.LossMask())
	}
}

func TestPackRejectsOverlongAndBadBlock(t *testing.T) {
	if _, err := Pack([][]int{{1, 2, 3}}, 2, 0); err == nil {
		t.Fatal("want error for sequence longer than block")
	}
	if _, err := Pack(nil, 0, 0); err == nil {
		t.Fatal("want error for non-positive blockSize")
	}
}

func TestPackSkipsEmpty(t *testing.T) {
	res, err := Pack([][]int{{1, 2}, {}, {3}}, 4, 0)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	if len(res.Blocks) != 1 || !eqInt(res.Blocks[0].SegmentLengths, []int{2, 1}) {
		t.Fatalf("empty not skipped: %+v", res.Blocks)
	}
}

// Invariant: the block-diagonal mask isolates segments — a position never
// attends across a segment boundary, never to the future, and never to/from pad.
func TestBlockMaskIsolatesSegments(t *testing.T) {
	res, err := Pack([][]int{{1, 2, 3}, {4, 5}}, 6, 0)
	if err != nil {
		t.Fatalf("Pack: %v", err)
	}
	b := res.Blocks[0] // segIDs: 0,0,0,1,1,-1
	mask := BlockMask(b)
	if err := mlx.Eval(mask); err != nil {
		t.Fatalf("eval: %v", err)
	}
	vals, err := mlx.ToSlice[float32](mask)
	if err != nil {
		t.Fatalf("toslice: %v", err)
	}
	n := len(b.SegmentID)
	open := func(i, j int) bool { return vals[i*n+j] == 0 }

	// Within segment 0 (positions 0..2): causal lower-triangle open.
	if !open(2, 0) || !open(2, 1) || !open(2, 2) {
		t.Fatal("intra-segment causal attention should be open")
	}
	// Future masked: position 0 must not attend to position 2.
	if open(0, 2) {
		t.Fatal("future attention should be masked")
	}
	// Cross-segment masked: position 3 (segment 1) must not attend to position
	// 0 (segment 0), even though 0 < 3.
	if open(3, 0) || open(3, 2) {
		t.Fatal("cross-segment attention should be masked")
	}
	// Segment 1 internal: position 4 attends to position 3.
	if !open(4, 3) {
		t.Fatal("intra-segment-1 attention should be open")
	}
	// Padding (position 5) attends to nothing and nothing attends to it.
	for j := range n {
		if open(5, j) || open(j, 5) {
			t.Fatalf("padding position 5 leaked attention at j=%d", j)
		}
	}
}

func TestErrorRateAndHardFilter(t *testing.T) {
	f := DefaultHardFilter() // MinTraceLen 5000, MinErrorRate 0.75
	cases := []struct {
		p    Problem
		keep bool
	}{
		{Problem{TraceLen: 6000, Rollouts: 8, Errors: 7}, true},  // 0.875 >= 0.75
		{Problem{TraceLen: 6000, Rollouts: 8, Errors: 6}, true},  // 0.75 >= 0.75
		{Problem{TraceLen: 6000, Rollouts: 8, Errors: 5}, false}, // 0.625 < 0.75
		{Problem{TraceLen: 4000, Rollouts: 8, Errors: 8}, false}, // too short
		{Problem{TraceLen: 6000, Rollouts: 0, Errors: 0}, false}, // no rollouts -> rate 0
	}
	for i, c := range cases {
		if got := f.Keep(c.p); got != c.keep {
			t.Fatalf("case %d: Keep = %v, want %v (rate %.3f)", i, got, c.keep, c.p.ErrorRate())
		}
	}
	all := make([]Problem, len(cases))
	for i, c := range cases {
		all[i] = c.p
	}
	if got := f.Select(all); len(got) != 2 {
		t.Fatalf("Select kept %d, want 2", len(got))
	}
}

// The LR schedule warms up linearly to Peak, then cosine-decays to Final.
func TestScheduleWarmupAndCosineDecay(t *testing.T) {
	s := Schedule{Peak: 5e-5, Final: 8e-8, WarmupSteps: 10, TotalSteps: 110}
	// Warmup: step 0 is Peak/10, step 9 is Peak.
	if got := s.LR(0); math.Abs(got-5e-5/10) > 1e-12 {
		t.Fatalf("LR(0) = %v, want %v", got, 5e-5/10)
	}
	if got := s.LR(9); math.Abs(got-5e-5) > 1e-12 {
		t.Fatalf("LR(9) = %v, want Peak", got)
	}
	// Just after warmup, near Peak; at the end, Final.
	if got := s.LR(10); got > 5e-5+1e-12 || got < 4.9e-5 {
		t.Fatalf("LR(10) = %v, want ~Peak", got)
	}
	if got := s.LR(110); math.Abs(got-8e-8) > 1e-9 {
		t.Fatalf("LR(end) = %v, want Final", got)
	}
	// Monotone non-increasing through the decay region.
	prev := s.LR(10)
	for i := 11; i <= 110; i++ {
		cur := s.LR(i)
		if cur > prev+1e-15 {
			t.Fatalf("LR not decreasing at step %d: %v > %v", i, cur, prev)
		}
		prev = cur
	}
}

// fakeTrainer records the stages and problem counts it was handed.
type fakeTrainer struct {
	stages   []string
	dataLens []int
}

func (f *fakeTrainer) TrainStage(ctx context.Context, stage Stage, inDir string, problems []Problem) (string, error) {
	f.stages = append(f.stages, stage.Name)
	f.dataLens = append(f.dataLens, len(problems))
	return inDir + "/" + stage.Name, nil
}

func TestCurriculumRunsStagesAndAppliesFilter(t *testing.T) {
	problems := []Problem{
		{ID: "easy", TraceLen: 6000, Rollouts: 8, Errors: 1}, // rate 0.125 -> filtered in hard stage
		{ID: "hard", TraceLen: 6000, Rollouts: 8, Errors: 7}, // rate 0.875 -> kept
	}
	cur := DefaultCurriculum3B(100, 40)
	tr := &fakeTrainer{}
	out, err := cur.Run(context.Background(), tr, "/ck/base", problems)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "/ck/base/broad/hard" {
		t.Fatalf("final dir = %q", out)
	}
	if !eqStr(tr.stages, []string{"broad", "hard"}) {
		t.Fatalf("stages = %v", tr.stages)
	}
	// Broad stage sees all problems; hard stage sees only the filtered subset.
	if tr.dataLens[0] != 2 {
		t.Fatalf("broad stage saw %d problems, want 2", tr.dataLens[0])
	}
	if tr.dataLens[1] != 1 {
		t.Fatalf("hard stage saw %d problems, want 1 (filtered)", tr.dataLens[1])
	}
}

func TestCurriculumNilTrainer(t *testing.T) {
	if _, err := (Curriculum{}).Run(context.Background(), nil, "/x", nil); err == nil {
		t.Fatal("want error for nil trainer")
	}
}

func eqInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func eqF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func eqStr(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
