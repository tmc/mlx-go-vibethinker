// Package long2short implements VibeThinker's Long2Short math-RL reward
// reshaping: a second math-RL stage that rewards brevity among correct traces
// without changing the group's reward baseline.
//
// For a prompt group, incorrect traces keep their rewards unchanged. Among the
// correct set C, each trace i of length Lᵢ gets a brevity score sᵢ = 1/Lᵢ.
// Let s̄ be the mean brevity over C. The reshaped reward is
//
//	r′ᵢ = rᵢ + λ·(sᵢ − s̄) / max_{j∈C} |sⱼ − s̄|,   λ = 0.2,
//
// where the normalizing max is taken over C only — incorrect traces have no
// brevity score. The shift is zero-sum over C, because Σ_C (sᵢ − s̄) = 0, so
//
//	Σ_C (r′ᵢ − rᵢ) = 0
//
// and the group's mean reward — hence the GRPO baseline — is unchanged. Shorter
// correct traces are nudged above the mean and longer ones below, steering the
// policy toward token efficiency while preserving the correct-vs-incorrect
// reward gap. If every correct trace has the same length the denominator is
// zero; the rewards are then left unchanged. These properties (zero-sum,
// baseline-preserving, equal-length no-op) are pinned by the tests.
package long2short
