// Package eval implements VibeThinker's Pass@1 benchmark harness (DESIGN §4.7).
//
// Pass@1 is averaged over k independent samples per prompt under
// vLLM-equivalent sampling, with strictly binary rewards. For a benchmark of
// prompts q ∈ Q, draw k completions cₖ(q) from the policy at the eval sampling
// params and score each with a verifier rₖ(q) ∈ {0,1}. Pass@1 is the mean over
// prompts of the mean over samples of the binary reward:
//
//	Pass@1 = (1/|Q|) Σ_q (1/k) Σ_{j=1}^{k} r_j(q),   r_j(q) ∈ {0,1}.
//
// This is the unbiased single-sample success probability estimated by k draws,
// not the any-of-k Pass@K of spectrum/probe.
//
// The paper's vLLM-equivalent sampling params (Params): top_p 0.95, top_k -1
// (disabled), with temperature 1.0 for math/knowledge and 0.6 for code; k is 64
// for math, 8 for code, 16 for knowledge. Two ready-made configs, MathParams
// and CodeParams, carry these defaults.
//
// Sampling and scoring are seams: a Sampler produces completions and an
// rl.Environment (the verifier) scores them. The Sampler can be a model decode
// loop, a fixture, or the in-package fakes used by the property tests. The
// verifier's reward is thresholded to {0,1} so non-binary environments still
// yield a Pass@1; the threshold defaults to 0.5.
package eval
