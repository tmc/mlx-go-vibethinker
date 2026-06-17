// Package vibethinker reproduces the VibeThinker post-training pipeline in Go
// on top of the mlx-go stack.
//
// VibeThinker is a family of small dense reasoning models (1.5B, 3B) post-trained
// under the Spectrum-to-Signal Principle (SSP): supervised fine-tuning builds a
// diverse "spectrum" of solutions (maximizing Pass@K), then reinforcement learning
// amplifies the correct "signal" with MaxEnt-Guided Policy Optimization (MGPO).
// This module implements the method described in the two technical reports
// (arXiv 2511.06221 for the 1.5B, arXiv 2606.16140 for the 3B); it does not
// redistribute the released weights.
//
// The pipeline decomposes into composable stages, each in its own package:
//
//   - spectrum/probe — Domain-Aware Diversity Probing (per-subdomain Pass@K
//     specialist selection).
//   - spectrum/fuse  — Expert Model Fusion (weighted parameter average).
//   - spectrum/sft   — curriculum supervised fine-tuning (3B two-stage).
//   - signal/mgpo    — MGPO: GRPO modulated by a max-entropy-deviation weight.
//   - signal/long2short — zero-sum length-aware reward reshaping (3B).
//   - signal/multidomain — sequential math, code, STEM RL (3B).
//   - distill        — offline self-distillation with learning-potential
//     filtering (3B).
//   - instruct       — Instruct RL with rule-based and rubric rewards (3B).
//   - reward         — verifiable reward sources (math-verify, code sandbox,
//     rubric).
//   - data           — dataset loading, synthesis seams, and 10-gram
//     decontamination.
//   - eval           — Pass@1 over k samples and CLR test-time scaling (3B).
//
// Dependencies that cannot run locally — frontier teacher models, the original
// proprietary corpora, and full-scale GPU compute — are exposed as explicit,
// pluggable seams rather than performed silently. See DESIGN.md for the full
// specification and the faithfulness/realizability contract.
package vibethinker
