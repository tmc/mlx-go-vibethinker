// Package probe implements VibeThinker's Domain-Aware Diversity Probing: the
// first step of Diversity-Exploring Distillation in the Spectrum (SFT) phase
// (DESIGN §4.1; correctness property 5 in DESIGN §5).
//
// A domain is partitioned into N subdomains (paper: math N = 4 =
// {algebra, geometry, calculus, statistics}; code and STEM surface their own
// partition as config). Each subdomain Sᵢ holds a probing set Dᵢ = {(q, a)}.
// During SFT, every k steps each checkpoint Mₜ is evaluated on every Dᵢ with
// Pass@K, and the per-subdomain specialist is selected for diversity:
//
//	Mᵢ* = argmaxₜ Pᵢ(t),
//
// the checkpoint with the highest Pass@K on subdomain i — not the lowest
// validation loss or the highest Pass@1. Selecting the most diverse checkpoint
// per subdomain builds the broad solution spectrum the later fusion step merges.
//
// Pass@K is the unbiased Chen et al. (HumanEval) estimator. For a query with n
// samples drawn and c correct,
//
//	pass@k = 1 − C(n−c, k) / C(n, k),
//
// evaluated through the numerically-stable product form
//
//	pass@k = 1 − Π_{i=n−c+1}^{n} (1 − k/i).
//
// The estimate is 1 when n−c < k (every size-k subset must hit a correct
// sample), so c = 0 ⇒ 0 and c = n ⇒ 1. See [PassK].
//
// Selection ([Select]) is a thin validating shell over an unexported argmax
// core; the model is injected as an [Evaluator] so the selector is tested
// without any real model. Properties pinned by the tests:
//
//   - PassK matches the combinatorial closed form C(n−c,k)/C(n,k) when k ≤ n.
//   - PassK is 0 at c = 0 and 1 at c = n.
//   - Select picks the highest-Pass@K checkpoint per subdomain, with ties
//     broken deterministically toward the earliest checkpoint.
package probe
