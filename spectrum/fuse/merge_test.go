package fuse

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

// f32 builds a small float32 model with one tensor "w" of the given values.
func f32(name string, vals []float32, shape ...int) Model {
	if len(shape) == 0 {
		shape = []int{len(vals)}
	}
	return Model{Name: name, Tensors: map[string]*mlx.Array{
		"w": mlx.NewArray(vals, shape...),
	}}
}

// slice evaluates a tensor and returns its float32 contents.
func slice(t *testing.T, a *mlx.Array) []float32 {
	t.Helper()
	if err := mlx.Eval(a); err != nil {
		t.Fatalf("eval: %v", err)
	}
	got, err := mlx.ToSlice[float32](a)
	if err != nil {
		t.Fatalf("toslice: %v", err)
	}
	return got
}

func approxEqual(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}

// Invariant (DESIGN §5.4): uniform merge of N identical models is the identity.
func TestUniformMergeOfIdenticalIsIdentity(t *testing.T) {
	vals := []float32{1, -2, 3.5, 0}
	models := []Model{
		f32("a", vals),
		f32("b", vals),
		f32("c", vals),
	}
	out, err := Merge(models, UniformWeights(3))
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if !approxEqual(slice(t, out["w"]), vals, 1e-6) {
		t.Fatalf("identity merge changed values: got %v want %v", slice(t, out["w"]), vals)
	}
}

// Invariant (DESIGN §5.4 / §3): weights summing to 1 preserve parameter
// magnitude — the merge is a convex combination, so it lies within the range of
// the inputs and cannot exceed their bound.
func TestWeightedAverageIsConvexCombination(t *testing.T) {
	a := f32("a", []float32{0, 10})
	b := f32("b", []float32{4, 2})
	// weights 0.25/0.75 -> 0.25*0 + 0.75*4 = 3 ; 0.25*10 + 0.75*2 = 4
	out, err := Merge([]Model{a, b}, []float64{0.25, 0.75})
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	want := []float32{3, 4}
	if !approxEqual(slice(t, out["w"]), want, 1e-5) {
		t.Fatalf("got %v want %v", slice(t, out["w"]), want)
	}
	// Magnitude preservation: every output is within [min,max] of the inputs.
	got := slice(t, out["w"])
	for i, g := range got {
		lo := math.Min(float64([]float32{0, 10}[i]), float64([]float32{4, 2}[i]))
		hi := math.Max(float64([]float32{0, 10}[i]), float64([]float32{4, 2}[i]))
		if float64(g) < lo-1e-5 || float64(g) > hi+1e-5 {
			t.Fatalf("output %v outside input range [%v,%v]", g, lo, hi)
		}
	}
}

// A weight of 1 on one model selects it exactly.
func TestWeightSelectsSingleModel(t *testing.T) {
	a := f32("a", []float32{1, 2, 3})
	b := f32("b", []float32{9, 9, 9})
	out, err := Merge([]Model{a, b}, []float64{1, 0})
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if !approxEqual(slice(t, out["w"]), []float32{1, 2, 3}, 1e-6) {
		t.Fatalf("got %v want first model", slice(t, out["w"]))
	}
}

// Invariant (DESIGN §5.4): shape mismatch fails closed.
func TestShapeMismatchFailsClosed(t *testing.T) {
	a := Model{Name: "a", Tensors: map[string]*mlx.Array{"w": mlx.NewArray([]float32{1, 2, 3, 4}, 2, 2)}}
	b := Model{Name: "b", Tensors: map[string]*mlx.Array{"w": mlx.NewArray([]float32{1, 2, 3, 4}, 4)}}
	if _, err := Merge([]Model{a, b}, UniformWeights(2)); err == nil {
		t.Fatal("want error on shape mismatch")
	}
}

// Invariant (DESIGN §5.4): name mismatch fails closed.
func TestNameMismatchFailsClosed(t *testing.T) {
	a := Model{Name: "a", Tensors: map[string]*mlx.Array{"w": mlx.NewArray([]float32{1}, 1)}}
	b := Model{Name: "b", Tensors: map[string]*mlx.Array{"v": mlx.NewArray([]float32{1}, 1)}}
	if _, err := Merge([]Model{a, b}, UniformWeights(2)); err == nil {
		t.Fatal("want error on name mismatch")
	}
}

// Invariant (DESIGN §5.4): dtype mismatch fails closed.
func TestDtypeMismatchFailsClosed(t *testing.T) {
	f := mlx.NewArray([]float32{1, 2}, 2)
	h := mlx.Astype(f, mlx.Float16)
	a := Model{Name: "a", Tensors: map[string]*mlx.Array{"w": f}}
	b := Model{Name: "b", Tensors: map[string]*mlx.Array{"w": h}}
	if _, err := Merge([]Model{a, b}, UniformWeights(2)); err == nil {
		t.Fatal("want error on dtype mismatch")
	}
}

func TestWeightValidation(t *testing.T) {
	a := f32("a", []float32{1})
	b := f32("b", []float32{2})
	cases := []struct {
		name    string
		weights []float64
		wantErr bool
	}{
		{"sum-not-one", []float64{0.5, 0.4}, true},
		{"negative", []float64{1.5, -0.5}, true},
		{"nan", []float64{math.NaN(), 1}, true},
		{"wrong-count", []float64{1}, true},
		{"valid", []float64{0.3, 0.7}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Merge([]Model{a, b}, tc.weights)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestEmptyModelsFailsClosed(t *testing.T) {
	if _, err := Merge(nil, nil); err == nil {
		t.Fatal("want error on empty models")
	}
}

func TestUniformWeights(t *testing.T) {
	w := UniformWeights(4)
	var sum float64
	for _, x := range w {
		if x != 0.25 {
			t.Fatalf("weight %v, want 0.25", x)
		}
		sum += x
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("sum %v, want 1", sum)
	}
	if UniformWeights(0) != nil {
		t.Fatal("UniformWeights(0) should be nil")
	}
}

// Round-trip the file-based shell: write three identical safetensors, fuse
// uniformly, reload, and confirm identity.
func TestMergeFilesRoundTrip(t *testing.T) {
	dir := t.TempDir()
	vals := []float32{1, 2, 3, 4}
	var paths []string
	for _, n := range []string{"a", "b", "c"} {
		p := filepath.Join(dir, n+".safetensors")
		if err := mlx.SaveSafetensors(p, map[string]*mlx.Array{"w": mlx.NewArray(vals, 2, 2)}, nil); err != nil {
			t.Fatalf("save %s: %v", n, err)
		}
		paths = append(paths, p)
	}
	out := filepath.Join(dir, "merged.safetensors")
	if err := MergeFiles(paths, UniformWeights(len(paths)), out); err != nil {
		t.Fatalf("MergeFiles: %v", err)
	}
	tensors, meta, err := mlx.LoadSafetensors(out)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !approxEqual(slice(t, tensors["w"]), vals, 1e-6) {
		t.Fatalf("round-trip changed values: %v", slice(t, tensors["w"]))
	}
	if meta["fused"] != "expert-model-fusion" {
		t.Fatalf("missing fusion metadata: %v", meta)
	}
}
