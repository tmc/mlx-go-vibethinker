// Package clr implements VibeThinker's Claim-Level Reliability (CLR) test-time
// scaling for the 3B model (DESIGN §4.7, §5.6).
//
// For one prompt, generate K trajectories. From each trajectory k extract M
// decision-relevant claims and a final answer, and self-verify each claim to
// v_{k,m} ∈ {0,1}. The trajectory's reliability is the mean claim validity
// raised to the M-th power:
//
//	r_k = ( (1/M) Σ_{m=1}^{M} v_{k,m} )^M.
//
// The exponent M makes the penalty for any flawed claim nonlinear: r_k = 1 iff
// every claim is valid, and a single invalid claim drops reliability sharply
// (M=5, 4/5 valid ⇒ 0.8^5 ≈ 0.328). Cluster the trajectories' final answers by
// equivalence; the selected answer is the cluster G maximizing the summed
// reliability of its members:
//
//	answer* = argmax_G Σ_{k∈G} r_k.
//
// Run this whole flow R times (paper: R=8) and report the mean Pass@1 of the
// selected answer against the gold answer.
//
// Claim extraction and self-verification are model-prompted steps and are left
// as a gated Verifier interface; the package ships a deterministic fake
// (FakeVerifier) for the property tests. Answer equivalence is supplied as an
// Equivalence function so a benchmark can plug in its own normalization.
package clr
