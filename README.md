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

Implemented. Every stage in [`DESIGN.md`](./DESIGN.md) is built on the mlx-go
stack, every §5 correctness invariant has a passing property test, and the full
1.5B and 3B recipes run end to end on a toy config (a tiny 2-layer Qwen2) without
NaNs, emitting a merged, RL-updated, and distilled checkpoint with provenance.
`DESIGN.md` remains the contract this reproduction is held to.

Run the toy pipeline (the synthetic model registry is behind the `modelir`
build tag):

    go test -tags modelir ./...                       # all invariants, -race
    go run  -tags modelir ./cmd/vibethinker-train -size 1.5b
    go run  -tags modelir ./cmd/vibethinker-train -size 3b
    go run  ./cmd/vibethinker-eval -mode pass1        # Pass@1 over a fake sampler
    go run  ./cmd/vibethinker-eval -mode clr          # CLR reliability selection

### Gates (not run locally; explicit seams with in-repo fakes)

- **Teacher** (`teacher`) — multi-path traces, pseudo-labels, CLR claim
  extraction/self-verification need a frontier model. Supply a real `Teacher`;
  the in-repo `Fake` drives tests.
- **Code sandbox** (`reward/sandbox`) — the local `ExecRunner` is fail-closed; a
  real run supplies an isolated `Runner`.
- **Rubric reward model** (`reward/rubric`) — a gated `Scorer` interface with a
  fake; a real run supplies a judge model.
- **Full-scale compute** — the multi-step weight optimizer (`TrainFullFineTune`)
  and GPU-scale training are documented compute gates; the toy stages exercise
  the loss graphs and every algorithmic transform, not a full optimizer loop.
- **Original corpora** — datasets are pluggable `Loader`/`Synthesizer` seams.

## License

MIT.
