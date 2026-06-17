// Package instruct implements VibeThinker's Instruct RL stage for the 3B model:
// on-policy RL on mixed instruction data that improves controllability without
// sacrificing reasoning (DESIGN §4.4).
//
// The reward is composed by prompt type. Explicit-constraint prompts are scored
// by rule-based validators that check verifiable properties of the response —
// required format, item count, ordering, keyword inclusion or exclusion, and
// completion. Open-ended prompts are scored by a rubric-based reward model,
// which is a gated seam (an injected rl.Environment) because it needs a judge
// model not bundled here.
//
// [Composer] routes each prompt to its reward source and exposes the result as
// a single rl.Environment, so the same MGPO/GRPO optimizer drives Instruct RL
// unchanged. The rule validators are pure and fully tested; the rubric path is
// exercised in tests with a fake judge.
package instruct
