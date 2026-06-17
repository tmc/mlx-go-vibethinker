# mlx-go-vibethinker

A Go reproduction of the **VibeThinker** post-training pipeline, built on the
[mlx-go](https://github.com/tmc/mlx-go) stack.

VibeThinker (Weibo) is a family of small dense reasoning models — 1.5B and 3B —
that reach frontier-level scores on verifiable math and coding benchmarks at a
fraction of the usual parameter count and training cost. Their method, the
**Spectrum-to-Signal Principle (SSP)**, runs supervised fine-tuning to build a
diverse *spectrum* of solutions (maximizing Pass@K), then reinforcement learning
to amplify the correct *signal* with **MaxEnt-Guided Policy Optimization
(MGPO)**.

This module reproduces the *method* described in the two technical reports. It
does not redistribute the released weights, and it does not pretend to perform
steps that require resources we cannot supply — frontier teacher models, the
original proprietary corpora, and full-scale GPU compute are exposed as
explicit, pluggable seams.

Source papers:

- VibeThinker-1.5B — "Tiny Model, Big Logic", arXiv 2511.06221 (base
  Qwen2.5-Math-1.5B).
- VibeThinker-3B — "Exploring the Frontier of Verifiable Reasoning", arXiv
  2606.16140 (base Qwen2.5-Coder-3B).

## What it implements

- **Spectrum (SFT):** Diversity-Exploring Distillation — Domain-Aware Diversity
  Probing (per-subdomain Pass@K specialist selection) and Expert Model Fusion
  (weighted parameter average). 3B adds two-stage curriculum SFT.
- **Signal (RL):** MGPO — GRPO with a max-entropy-deviation advantage weight
  `w_ME = exp(-λ·D_ME(p_c‖0.5))`. 3B adds multi-domain RL (math→code→STEM), a
  single 64K context, and Long2Short token-efficiency RL.
- **3B consolidation:** offline self-distillation (learning-potential
  filtering) and Instruct RL (rule + rubric rewards).
- **Evaluation:** Pass@1 over k samples with vLLM-equivalent sampling, plus CLR
  (Claim-Level Reliability) test-time scaling.
- **Data:** 10-gram decontamination, quality filtering, and synthesis seams.

## Status

Pre-implementation. [`DESIGN.md`](./DESIGN.md) is the specification and the
contract this reproduction is held to; it records the faithful mapping of every
paper step onto a concrete mlx-go primitive and marks each external dependency
as an explicit gate. The design has been cross-checked against the papers and
the mlx-go reference sources.

## License

MIT.
