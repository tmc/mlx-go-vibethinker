// Package rubric implements VibeThinker's rubric-based reward model (DESIGN
// §4.4/§4.5: open-ended Instruct-RL prompts are scored by a "rubric-based
// reward model" — a gated interface).
//
// A rubric reward grades an open-ended completion against a textual rubric and
// returns a scalar in [0,1] —
//
//	r(q, y) = Scorer.Score(rubric(q), y) ∈ [0, 1].
//
// The scoring model is a frontier reward/judge model we cannot run locally, so
// it sits behind a Scorer interface (a gate) rather than being bundled:
//
//   - FakeScorer is an in-repo, deterministic Scorer used by tests and toy
//     pipelines. It returns a fixed score or a per-input function — no model
//     call, no network.
//   - A real reward model is a documented gate: GatedScorer reports the model
//     it would call and fails closed (ErrScorerGated) until a concrete Scorer
//     is supplied. This package ships no real model.
//
// Environment adapts a Scorer plus a per-prompt Rubric source to mlx-go-rl's
// Environment, clamping the returned score into [0,1], so the rubric reward
// composes with the MGPO/GRPO optimizer used for Instruct RL.
package rubric
