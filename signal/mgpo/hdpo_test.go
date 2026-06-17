package mgpo

import (
	"math"
	"testing"
)

// Phase-C property (DESIGN_RL_UPGRADE.md §2 Tier 3, HDPO): with LambdaJSD = 0 the
// cliff-JSD term contributes nothing and HDPOLossTerm equals the base GRPO loss
// bit-for-bit, even when cliff groups are present — the off-path is the
// unmodified baseline. The control half (LambdaJSD > 0 with a cliff group and a
// nonzero JSD) must change the loss, proving the knob is live.
func TestHDPOZeroLambdaBitIdenticalToBaseline(t *testing.T) {
	const base = 0.42
	rewards := [][]float64{
		{1, 0, 1, 1}, // learnable
		{0, 0, 0, 0}, // cliff (all fail)
	}
	jsd := []float64{0, 0.3} // nonzero JSD on the cliff group

	// LambdaJSD = 0: bit-identical to base, despite the cliff group + nonzero JSD.
	off, err := HDPOLossTerm(base, rewards, jsd, HDPOConfig{})
	if err != nil {
		t.Fatalf("HDPOLossTerm off: %v", err)
	}
	if off != base {
		t.Fatalf("LambdaJSD=0 gave %v, want base %v bit-for-bit", off, base)
	}

	// Control: LambdaJSD > 0 with a cliff group + nonzero JSD must differ.
	on, err := HDPOLossTerm(base, rewards, jsd, HDPOConfig{LambdaJSD: 0.5})
	if err != nil {
		t.Fatalf("HDPOLossTerm on: %v", err)
	}
	if on == base {
		t.Fatalf("LambdaJSD=0.5 left the loss at base %v; cliff-JSD knob is inert", base)
	}
	// The added term is exactly λ · mean(JSD over cliff) = 0.5 * 0.3.
	want := base + 0.5*0.3
	if math.Abs(on-want) > 1e-12 {
		t.Fatalf("HDPO loss = %v, want base + λ·meanJSD = %v", on, want)
	}
}

// Phase-C property: HDPO is gated entirely on the cliff set. A batch with no
// zero-reward group gets no HDPO contribution even at LambdaJSD > 0, and the
// cliff set is exactly the all-fail groups.
func TestHDPOCliffSetGating(t *testing.T) {
	rewards := [][]float64{
		{1, 0, 1, 1}, // acc 0.75 — not cliff
		{0, 0, 0, 0}, // acc 0    — cliff
		{1, 1, 1, 1}, // acc 1    — not cliff (this is a "ceiling", not a cliff)
		{0, 0, 0, 1}, // acc 0.25 — not cliff
	}
	cliff := CliffSet(rewards)
	if len(cliff) != 1 || cliff[0] != 1 {
		t.Fatalf("CliffSet = %v, want [1] (only the all-fail group)", cliff)
	}

	// A batch with no cliff group: HDPO adds nothing even at λ>0.
	noCliff := [][]float64{{1, 0, 1, 1}, {1, 1, 1, 1}}
	jsd := []float64{0.9, 0.9}
	got, err := HDPOLossTerm(1.0, noCliff, jsd, HDPOConfig{LambdaJSD: 2.0})
	if err != nil {
		t.Fatalf("HDPOLossTerm no-cliff: %v", err)
	}
	if got != 1.0 {
		t.Fatalf("no-cliff batch got HDPO contribution: %v != base 1.0", got)
	}
}

// Phase-C property: the JSD self-teacher term is a proper Jensen-Shannon
// divergence — zero for identical distributions, symmetric, non-negative, and
// bounded by ln 2.
func TestHDPOJSDProperties(t *testing.T) {
	p := []float64{0.7, 0.2, 0.1}
	q := []float64{0.1, 0.3, 0.6}

	// Identical distributions ⇒ JSD = 0.
	same, err := JSD(p, p)
	if err != nil {
		t.Fatalf("JSD(p,p): %v", err)
	}
	if math.Abs(same) > 1e-12 {
		t.Fatalf("JSD(p,p) = %v, want 0", same)
	}

	// Symmetry.
	pq, err := JSD(p, q)
	if err != nil {
		t.Fatalf("JSD(p,q): %v", err)
	}
	qp, err := JSD(q, p)
	if err != nil {
		t.Fatalf("JSD(q,p): %v", err)
	}
	if math.Abs(pq-qp) > 1e-12 {
		t.Fatalf("JSD asymmetric: %v != %v", pq, qp)
	}

	// Bounds: 0 < JSD ≤ ln 2 for distinct distributions.
	if pq <= 0 || pq > math.Ln2+1e-12 {
		t.Fatalf("JSD(p,q) = %v out of (0, ln2]", pq)
	}

	// Accepts unnormalized (top-k weight) vectors and normalizes internally:
	// scaling a distribution does not change the divergence.
	scaled := []float64{7, 2, 1} // = 10*p
	js2, err := JSD(scaled, q)
	if err != nil {
		t.Fatalf("JSD(scaled,q): %v", err)
	}
	if math.Abs(js2-pq) > 1e-12 {
		t.Fatalf("JSD not scale-invariant: %v != %v", js2, pq)
	}

	// Disjoint supports reach the ln 2 ceiling.
	full, err := JSD([]float64{1, 0}, []float64{0, 1})
	if err != nil {
		t.Fatalf("JSD disjoint: %v", err)
	}
	if math.Abs(full-math.Ln2) > 1e-12 {
		t.Fatalf("JSD of disjoint supports = %v, want ln2 %v", full, math.Ln2)
	}
}

// Phase-C cross-cutting invariant: HDPO is a loss-side addition gated on the
// cliff set; it never touches the advantage path, so the MGPO no-op rule and the
// advantage computation are entirely unaffected by HDPOConfig. We confirm the
// advantage of a cliff group is unchanged (still all-zero) — HDPO does not
// resurrect a gradient via the advantage, only via the added JSD loss term.
func TestHDPODoesNotTouchAdvantage(t *testing.T) {
	cliff := [][]float64{{0, 0, 0, 0}}
	adv, err := ScaledAdvantagesOpt(cliff, 1.0, Options{})
	if err != nil {
		t.Fatalf("ScaledAdvantagesOpt: %v", err)
	}
	for j, a := range adv[0] {
		if a != 0 {
			t.Fatalf("cliff group advantage slot %d = %v, want 0 (HDPO must not alter the advantage)", j, a)
		}
	}
}
