package mgpo

import (
	"math"
	"testing"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// Invariant (DESIGN §5.2): D_ME(0.5) = 0.
func TestDMEZeroAtHalf(t *testing.T) {
	if d := DME(0.5); math.Abs(d) > 1e-12 {
		t.Fatalf("DME(0.5) = %v, want 0", d)
	}
}

// Invariant (DESIGN §5.2): w_ME peaks at p_c = 0.5 and decays monotonically
// toward p_c in {0,1}.
func TestWeightPeaksAtHalfAndDecays(t *testing.T) {
	const lambda = 1.0
	half, _ := Weight(lambda, 0.5)
	if math.Abs(half-1.0) > 1e-12 {
		t.Fatalf("w_ME(0.5) = %v, want 1", half)
	}
	// Monotone decay on [0.5, 1]: weight strictly decreasing as p_c rises.
	prev := half
	for _, pc := range []float64{0.6, 0.7, 0.8, 0.9, 0.99, 1.0} {
		w, _ := Weight(lambda, pc)
		if w >= prev {
			t.Fatalf("w_ME not decreasing past 0.5: w(%.2f)=%v >= prev %v", pc, w, prev)
		}
		if w <= 0 || w > 1 {
			t.Fatalf("w_ME(%.2f) = %v out of (0,1]", pc, w)
		}
		prev = w
	}
	// Symmetry: D_ME and hence w_ME are symmetric about 0.5.
	for _, d := range []float64{0.1, 0.25, 0.4} {
		wlo, _ := Weight(lambda, 0.5-d)
		whi, _ := Weight(lambda, 0.5+d)
		if math.Abs(wlo-whi) > 1e-12 {
			t.Fatalf("w_ME asymmetric: w(%.2f)=%v w(%.2f)=%v", 0.5-d, wlo, 0.5+d, whi)
		}
	}
}

// Invariant (DESIGN §5.1): λ=0 ⇒ w_ME=1 for all p_c.
func TestWeightOneAtLambdaZero(t *testing.T) {
	for _, pc := range []float64{0, 0.01, 0.3, 0.5, 0.7, 0.99, 1} {
		w, err := Weight(0, pc)
		if err != nil {
			t.Fatalf("Weight(0,%v): %v", pc, err)
		}
		if w != 1.0 {
			t.Fatalf("w_ME(λ=0, p_c=%v) = %v, want exactly 1", pc, w)
		}
	}
}

// D_ME is non-negative everywhere and approaches log 2 at the extremes.
func TestDMENonNegativeAndBounded(t *testing.T) {
	for _, pc := range []float64{0, 0.001, 0.2, 0.5, 0.8, 0.999, 1} {
		d := DME(pc)
		if d < 0 {
			t.Fatalf("DME(%v) = %v < 0", pc, d)
		}
		if d > math.Ln2+1e-9 {
			t.Fatalf("DME(%v) = %v exceeds ln2", pc, d)
		}
	}
	// At the extremes it should be very close to ln2.
	if math.Abs(DME(0)-math.Ln2) > 1e-6 {
		t.Fatalf("DME(0) = %v, want ~ln2", DME(0))
	}
}

func TestWeightRejectsBadLambda(t *testing.T) {
	if _, err := Weight(-0.1, 0.5); err == nil {
		t.Fatal("want error for negative lambda")
	}
	if _, err := Weight(math.Inf(1), 0.5); err == nil {
		t.Fatal("want error for infinite lambda")
	}
}

func TestAccuracy(t *testing.T) {
	cases := []struct {
		r    []float64
		want float64
	}{
		{[]float64{1, 1, 0, 0}, 0.5},
		{[]float64{1, 1, 1, 1}, 1.0},
		{[]float64{0, 0, 0}, 0.0},
		{nil, 0.0},
		{[]float64{0.7, 0, 0.3, 0}, 0.5}, // any reward > 0 counts as success
	}
	for _, c := range cases {
		if got := Accuracy(c.r); got != c.want {
			t.Fatalf("Accuracy(%v) = %v, want %v", c.r, got, c.want)
		}
	}
}

// Invariant (DESIGN §5.1, the central MGPO claim): at λ=0 the scaled advantages
// are bit-identical to plain rl.GroupAdvantage.
func TestScaledAdvantagesIdenticalToGRPOAtLambdaZero(t *testing.T) {
	rewards := [][]float64{
		{1, 0, 1, 0},
		{1, 1, 1, 0},
		{0, 0, 0, 1},
	}
	want := rl.GroupAdvantage(rewards)
	got, err := ScaledAdvantages(rewards, 0)
	if err != nil {
		t.Fatalf("ScaledAdvantages: %v", err)
	}
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("group %d rollout %d: got %v want %v (must be bit-identical)", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// At λ>0 the weight multiplies the normalized advantage by exactly w_ME(p_c).
func TestScaledAdvantagesAppliesWeight(t *testing.T) {
	rewards := [][]float64{{1, 1, 0, 0}} // p_c = 0.5 -> w_ME = 1 regardless of λ
	got, err := ScaledAdvantages(rewards, 5.0)
	if err != nil {
		t.Fatalf("ScaledAdvantages: %v", err)
	}
	base := rl.GroupAdvantage(rewards)
	for j := range base[0] {
		if math.Abs(got[0][j]-base[0][j]) > 1e-12 {
			t.Fatalf("p_c=0.5 should be unscaled: got %v want %v", got[0][j], base[0][j])
		}
	}

	// A group with p_c far from 0.5 gets advantages shrunk by w_ME < 1.
	rewards2 := [][]float64{{1, 1, 1, 0}} // p_c = 0.75
	pc := Accuracy(rewards2[0])
	w := math.Exp(-2.0 * DME(pc))
	got2, _ := ScaledAdvantages(rewards2, 2.0)
	base2 := rl.GroupAdvantage(rewards2)
	for j := range base2[0] {
		want := w * base2[0][j]
		if math.Abs(got2[0][j]-want) > 1e-9 {
			t.Fatalf("rollout %d: got %v want w·A=%v (w=%v)", j, got2[0][j], want, w)
		}
	}
}

func TestScaledAdvantagesRejectsBadLambda(t *testing.T) {
	if _, err := ScaledAdvantages([][]float64{{1, 0}}, -1); err == nil {
		t.Fatal("want error for negative lambda")
	}
}

func TestFlattenAndTensor(t *testing.T) {
	adv := [][]float64{{1, 2}, {3}}
	flat := FlattenAdvantages(adv)
	if len(flat) != 3 || flat[0] != 1 || flat[2] != 3 {
		t.Fatalf("flatten = %v, want [1 2 3]", flat)
	}
	arr, err := AdvantageTensor(flat)
	if err != nil {
		t.Fatalf("AdvantageTensor: %v", err)
	}
	shape := arr.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 1 {
		t.Fatalf("tensor shape = %v, want [3 1]", shape)
	}
	if _, err := AdvantageTensor(nil); err == nil {
		t.Fatal("want error for empty advantage slice")
	}
}

// End-to-end loss-level identity: at λ=0, mgpo.Loss equals rl.GRPOLoss with
// plain GroupAdvantage on the same toy log-prob tensors, bit-for-bit.
func TestLossIdenticalToGRPOAtLambdaZero(t *testing.T) {
	// 4 sequences (one group of 4), 3 tokens each.
	const seqs, toks = 4, 3
	mk := func(seed float32) *mlx.Array {
		vals := make([]float32, seqs*toks)
		for i := range vals {
			vals[i] = seed + float32(i)*0.01
		}
		return mlx.NewArray(vals, seqs, toks)
	}
	current := mk(-0.5)
	old := mlx.StopGradient(mk(-0.4))
	ref := mlx.StopGradient(mk(-0.45))
	maskVals := make([]float32, seqs*toks)
	for i := range maskVals {
		maskVals[i] = 1
	}
	mask := mlx.NewArray(maskVals, seqs, toks)
	rewards := [][]float64{{1, 0, 1, 0}}
	cfg := rl.DefaultGRPOConfig()

	// MGPO loss at λ=0.
	mgpoLoss, err := Loss(current, old, ref, mask, rewards, 0, cfg)
	if err != nil {
		t.Fatalf("mgpo.Loss: %v", err)
	}
	// Reference plain-GRPO loss with identical advantages.
	baseAdv := FlattenAdvantages(rl.GroupAdvantage(rewards))
	baseArr, _ := AdvantageTensor(baseAdv)
	grpoLoss, err := rl.GRPOLoss(current, old, ref, baseArr, mask, cfg)
	if err != nil {
		t.Fatalf("rl.GRPOLoss: %v", err)
	}
	if err := mlx.Eval(mgpoLoss, grpoLoss); err != nil {
		t.Fatalf("eval: %v", err)
	}
	a := mlx.ArrayItemFloat32(mgpoLoss)
	b := mlx.ArrayItemFloat32(grpoLoss)
	if a != b {
		t.Fatalf("mgpo λ=0 loss %v != grpo loss %v (must be bit-identical)", a, b)
	}
}
