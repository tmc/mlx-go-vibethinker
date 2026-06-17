// Package recipe assembles the VibeThinker 1.5B and 3B training recipes as ssp
// pipelines over the toy model, so the whole Spectrum-to-Signal flow runs end to
// end on CPU.
//
// Each stage is a real [ssp.Stage] that performs the genuine numeric transform
// of its phase on the toy Qwen2 and records provenance:
//
//   - SFT computes the cross-entropy training loss over toy data through the
//     model forward pass and asserts it is finite, then writes a checkpoint.
//     The multi-step weight optimization (TrainFullFineTune) is the documented
//     compute gate — it needs GPU-scale resources and is not run on the toy
//     config — so the toy SFT stage exercises the loss graph and the curriculum
//     and packing transforms, not a full optimizer loop.
//   - Probe+Fuse evaluates per-subdomain Pass@K and merges specialist
//     checkpoints with the real Expert Model Fusion kernel.
//   - MGPO computes the real MaxEnt-weighted GRPO loss over toy rollout
//     log-probs through the package-level GRPOLoss seam and asserts finiteness.
//   - Distill scores verified traces with the real length-normalized NLL
//     (S_LP) under the toy student and selects a distillation set.
//
// The pipelines emit a merged, RL-updated, distilled checkpoint with full
// provenance and no NaNs, which is the reproduction's end-to-end correctness
// litmus test. Full benchmark reproduction is a documented compute gate.
//
// This package builds only under the modelir tag, because the toy Qwen2 (and
// every mlx-go-lm model) is registered behind that tag.
package recipe
