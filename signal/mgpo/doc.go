// Package mgpo implements MaxEnt-Guided Policy Optimization, the Signal-phase
// RL objective of VibeThinker.
//
// MGPO is GRPO with one addition: each prompt group's advantage is scaled by a
// max-entropy-deviation weight that focuses learning on prompts whose empirical
// success rate is near 1/2 — the regime of maximum exploratory value — and
// down-weights prompts the policy already solves or always fails.
//
// For a prompt q, sample G rollouts from the old policy, score each with a
// verifiable reward to a binary rᵢ, and let the empirical accuracy be
//
//	p_c(q) = (1/G) Σ I(rᵢ = 1).
//
// The max-entropy deviation is the Bernoulli KL of p_c to p₀ = 1/2:
//
//	D_ME = p_c·log(p_c/p₀) + (1−p_c)·log((1−p_c)/(1−p₀)),   D_ME ≥ 0,
//
// and the advantage weight is
//
//	w_ME(p_c) = exp(−λ·D_ME),   λ ≥ 0,   w_ME ∈ (0, 1].
//
// w_ME peaks at 1 when p_c = 1/2 (D_ME = 0) and decays monotonically toward the
// extremes. At λ = 0, w_ME ≡ 1 and MGPO is exactly GRPO — a property the tests
// pin bit-for-bit. (The 1.5B paper writes this coefficient λ; the 3B paper
// writes the identical coefficient γ. We use λ.)
//
// The seam into the substrate is deliberate. mlx-go-rl's GroupAdvantage
// normalizes each group by its standard deviation, so scaling the raw rewards
// by a per-group factor cancels — (w·r − w·μ)/(w·σ) = (r − μ)/σ — and would be a
// no-op. The weight must therefore multiply the normalized advantage A, not the
// reward, matching the paper's A′ⱼ(q) = w_ME(p_c(q))·Aⱼ(q). Because w_ME is
// constant within a group, this is well defined per group. The scaled
// advantages are then fed to the package-level rl.GRPOLoss, which takes
// advantages as an explicit argument — unlike the GRPOEstimator methods, which
// compute advantages internally and expose no injection point.
//
// [Options] selects two opt-in, scale-safe post-GRPO refinements
// (DESIGN_RL_UPGRADE.md §2 Tier 1): Dr.GRPO advantage debiasing (drop the std
// divisor) and DAPO Clip-Higher (asymmetric PPO clip). The zero Options is the
// DESIGN.md baseline, so [ScaledAdvantages] and [Loss] are unchanged; use
// [ScaledAdvantagesOpt] and [LossOpt] to enable a refinement. Under either
// advantage normalization w_ME still multiplies the advantage, never the raw
// reward, so the no-op rule holds.
//
// [ScaledAdvantagesStep] adds DCPO Smooth Advantage Standardization (SAS,
// DESIGN_RL_UPGRADE.md §2 Tier 2): it standardizes each prompt's advantage over
// a cumulative per-prompt history carried in a [PromptStats] store across
// training steps, picking the minimum-magnitude of two smoothed estimates. A nil
// store (or the first visit of a fresh one) reproduces the plain advantage
// exactly, and w_ME still multiplies the smoothed advantage — the no-op rule is
// preserved. SAS composes with Dr.GRPO: it smooths whichever advantage
// normalization the options select.
//
// [DynamicSample] is DAPO Dynamic Sampling (DESIGN_RL_UPGRADE.md §2 Tier 2): a
// data-layer filter that drops prompt groups with accuracy 0 or 1 (zero
// group-relative advantage) before the loss, unifying with the std=0 guard. It
// only removes whole zero-gradient groups, so a retained group's advantage and
// loss are unchanged.
//
// [LossFRPO] is the FRPO Future-KL sibling loss (DESIGN_RL_UPGRADE.md §2 Tier 3).
// The substrate rl.GRPOLoss has no per-token injection point, so LossFRPO clones
// its surrogate and adds a reverse-cumulative-sum future-KL term on the per-token
// log-ratio. With a zero FRPOConfig it is bit-identical to rl.GRPOLoss;
// [LossFRPOScaled] wires it to the w_ME-scaled advantage, preserving the no-op
// rule.
package mgpo
