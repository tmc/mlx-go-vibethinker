# mlx-go-vibethinker — Design Spec

Reproduce the **VibeThinker** post-training pipeline in Go on top of the
`mlx-go` stack. The goal is a faithful, runnable reproduction of the
*method* described in the two technical reports — not the released weights.
Where a step needs a frontier teacher model or a GPU cluster we cannot
provide, the design exposes it as an explicit, pluggable **gate** rather than
pretending to perform it.

- VibeThinker-1.5B: "Tiny Model, Big Logic", arXiv 2511.06221 (base
  Qwen2.5-Math-1.5B).
- VibeThinker-3B: "Exploring the Frontier of Verifiable Reasoning", arXiv
  2606.16140 (base Qwen2.5-Coder-3B).

Status: **v0.1 design — pre-implementation.** This file is the contract the
NotebookLM review loop is run against. Hyperparameters in `[brackets]` are
taken verbatim from the papers; anything marked *(inferred)* is a design
choice the papers do not pin down.

---

## 1. Scope and non-goals

In scope (the reproducible method):

1. **SSP** — the Spectrum-to-Signal Principle that structures the whole
   pipeline: SFT builds a diverse solution *spectrum* (maximize Pass@K), RL
   amplifies the correct *signal*.
2. **Spectrum phase (SFT)** — Diversity-Exploring Distillation:
   Domain-Aware Diversity Probing → Expert Model Fusion (parameter merge).
   3B adds: data synthesis/quality control + two-stage curriculum SFT.
3. **Signal phase (RL)** — MaxEnt-Guided Policy Optimization (MGPO): GRPO
   with an entropy-deviation advantage weight `w_ME = exp(-λ·D_ME(p_c‖0.5))`.
   3B adds: multi-domain RL (math→code→STEM), single 64K context,
   Long2Short math RL.
4. **3B consolidation** — Offline Self-Distillation (learning-potential
   filtering) and Instruct RL (rule + rubric rewards).
5. **Evaluation** — Pass@1 over k samples with vLLM-equivalent sampling, plus
   CLR (Claim-Level Reliability) test-time scaling for the 3B.
6. **Data decontamination** — 10-gram overlap filtering vs eval sets.

Non-goals / external gates (cannot be done locally, must be plugged in):

- **Teacher generation** — multi-path reasoning traces and pseudo-labels come
  from "strong teacher models". We define a `Teacher` interface; a real run
  supplies an API-backed or local large model. Not bundled.
- **Pretraining** — we start from the published Qwen2.5 base weights.
- **Full-scale compute** — 3.9K (1.5B) / larger (3B) H800 GPU-hours. The code
  must *run* end-to-end on a tiny config (toy model, few steps) for
  correctness; full reproduction is a documented compute gate.
- **Exact training corpora** — proprietary synthetic data is not released. We
  target the public-dataset + synthesis recipe and make datasets pluggable.

The litmus test for "reproduction": every algorithmic transform in the papers
is implemented and unit-tested on toy tensors; every data/teacher/compute
dependency is an explicit seam, not a silent omission.

---

## 2. Substrate: what we build on (mlx-go)

Verified packages on disk (real import paths + symbols):

| Capability | Package | Key symbols |
|---|---|---|
| Arrays, autograd, VJP | `github.com/tmc/mlx-go/mlx` | `Array`, `Add/Mul/Matmul`, `Eval`, grad via `mlx/compile` |
| NN layers | `…/mlx/nn` | `Linear`, `Attention`, norms, activations |
| Optimizers | `…/mlx/optimizer` | `AdamW`, `Adam`, `SGD` |
| Graph/train-step compile | `…/mlx/compile` | compiled step |
| Weight I/O | `…/mlx-go/safetensors` | safetensors load/save |
| Model loading + arch registry | `mlx-go-lm/mlxlm`, `…/llm/models` | `Load`, `qwen2.gen.go` (Qwen2), `Register`, `LanguageModel` |
| SFT training loop | `mlx-go-lm/mlxlm/llm/training` | `ComputeLoss`, `ComputeLossFromLogits`, `BatchIterator`, `CompiledAdam`, `AccumulatedTrainingStep`, full fine-tune |
| Decode / rollouts | `mlx-go-lm/mlxlm/llm/decode` | `TokenIterator`, batch generator |
| Sampling | `mlx-go-lm/mlxlm/llm/sample` | temperature, top-p, top-k, repetition penalty |
| **GRPO** | `mlx-go-examples/mlx-go-rl` | package-level `GRPOLoss(current, old, ref, advantages, mask, config)` (advantages passed in — MGPO seam), `GroupAdvantage`/`GroupAdvantageDrGRPO` (`[][]float64`), `ImportanceWeights`, `KLDivergence`, `CorrectedLoss`, `EntropyController`, `StalenessFilter`, `Environment`/`EnvFromVerifyFunc`, `GRPOTrainer` (`Forward`/`Step`/`Apply`). Note: `GRPOEstimator.GRPOLoss*` *methods* compute advantages internally — not the MGPO seam. |
| Distillation | `mlx-go-examples/mlx-go-distill` | `OnlineDistiller`, KL/MSE loss |
| Merge (stub today) | `mlx-go-examples/mlx-go-rlm` | doc-only — merge logic to be built here |

Implication: **MGPO is a thin extension of the existing GRPO estimator**, the
SFT loop already exists, and model merging is the main genuinely-new numeric
kernel (and it is just a weighted parameter average over safetensors).

---

## 3. Package topology (this module)

`github.com/tmc/mlx-go-vibethinker`

```
vibethinker/                 # umbrella doc.go only; no logic
  ssp/                       # Spectrum-to-Signal orchestration types
    pipeline.go              # Stage interface, Pipeline runner, checkpoint provenance
  spectrum/                  # SFT "Spectrum" phase
    probe/                   # Domain-Aware Diversity Probing
      passk.go               # Pass@K estimator over a probing set
      probe.go               # periodic checkpoint eval -> per-domain specialist selection
    fuse/                    # Expert Model Fusion
      merge.go               # weighted parameter average over safetensors (w_i sum to 1)
    sft/                     # curriculum SFT driver (wraps mlxlm training)
      curriculum.go          # 3B two-stage (broad coverage -> hard-reasoning) gate
  signal/                    # RL "Signal" phase
    mgpo/                    # MaxEnt-Guided Policy Optimization
      weight.go              # D_ME (KL to 0.5) + w_ME = exp(-λ·D_ME); degrades to GRPO at λ=0
      mgpo.go                # wraps mlx-go-rl GRPOEstimator, injects w_ME into advantages
    long2short/              # zero-sum length-aware reward reshaping for correct traces
      reshape.go
    multidomain/             # sequential math->code->STEM RL driver (3B)
  distill/                   # Offline Self-Distillation (3B)
    potential.go             # learning-potential score S_LP (len-normalized NLL under student)
    select.go                # per-domain length-bucket selection, outlier trimming
  instruct/                  # Instruct RL (3B): rule-based + rubric reward composition
  reward/                    # verifiable reward sources
    mathverify/              # final-answer equivalence (rule-based)
    sandbox/                 # code execution reward (interface + local runner)
    rubric/                  # rubric-based reward model interface (gated)
  teacher/                   # Teacher interface (multi-path distillation, pseudo-labels) — gated
  data/                      # dataset loading + synthesis/expansion seams; parquet eval sets
    decontam/                # 10-gram overlap decontamination
  eval/                      # Pass@1 over k samples; benchmark harness
    clr/                     # Claim-Level Reliability test-time scaling (3B)
  internal/…                 # fakes, toy model/tokenizer for tests
cmd/
  vibethinker-train/         # drives a full SSP run from a config
  vibethinker-eval/          # runs eval/CLR
```

Naming follows the repo convention (explicit, lowercase, no `utils`). Each
package owns one concern; `ssp` is the only orchestrator.

Delegation seam (per the package-dev playbook): each exported entrypoint
(`Merge`, `Weight`, `Reshape`, `Score`, `Probe`) is a thin validating shell
over an unexported core that holds the numerics and can be regenerated.

---

## 4. The pipeline, stage by stage

### 4.0 SSP orchestration (`ssp`)

A `Stage` consumes a checkpoint + config and produces a new checkpoint with
recorded provenance (which stage, which inputs). A `Pipeline` is an ordered
list of stages with typed artifacts between them. This makes the 1.5B and 3B
recipes two different stage lists over the same components.

- **1.5B recipe:** SFT-spectrum (probe+fuse) → MGPO math (16K→32K) → MGPO code.
- **3B recipe:** curriculum SFT stage1 (broad) → SFT stage2 (hard) → [probe+fuse
  applied within each SFT stage] → MGPO math (64K, accuracy) → Long2Short math →
  MGPO code → MGPO STEM → offline self-distill → Instruct RL.

### 4.1 Spectrum phase — Diversity-Exploring Distillation (`spectrum`)

**Domain-Aware Diversity Probing (`spectrum/probe`).**
Partition a domain into `N` subdomains (paper: math `N=4` =
{algebra, geometry, calculus, statistics}). For each subdomain `Sᵢ` hold a
probing set `Dᵢ = {(q, a)}`. During SFT, every `k` steps evaluate the
checkpoint `Mₜ` on each `Dᵢ` with **Pass@K** and pick the per-subdomain
specialist `Mᵢ* = argmaxₜ Pᵢ(t)` — selecting for *diversity*, not lowest val
loss or highest Pass@1.

- `Pass@K`: draw K samples per probe query at the eval sampling params, score
  with the domain verifier, success if any sample passes. Estimator core uses
  the standard unbiased combinatorial form when `n_samples > K`.

**Expert Model Fusion (`spectrum/fuse`).**
Merge specialists into one SFT model by **weighted linear parameter average**:
`M_merge = Σ wᵢ·Mᵢ*`, with `wᵢ ≥ 0`, `Σ wᵢ = 1` (paper default: uniform
`wᵢ = 1/N`). Implemented as a per-tensor average across safetensors with
identical architecture/shape; the shell validates name/shape/dtype agreement
and weight normalization before delegating to the averaging core.

**Curriculum SFT (`spectrum/sft`, 3B).** Two stages:
- Stage 1 — broad coverage / cold start over the full quality-filtered set.
  `[global batch 128, LR 5e-5 cosine→8e-8, 5 epochs, 5% linear warmup,
  sequence packing]`. *Realizability:* `mlx-go-lm` training supplies
  `TrainFullFineTune`, `ComputeLoss`, `BatchIterator`, grad accumulation, and a
  per-step `lr *mlx.Array` input (so cosine→8e-8 with 5% warmup is a CPU-side
  schedule we feed in). **Sequence packing is not provided** (only padding
  iterators) and must be built in `spectrum/sft` as a preprocessing step that
  concatenates sequences and builds the block-diagonal attention mask.
- Stage 2 — hard-reasoning subset from stage-1 checkpoint. Filter:
  `[drop traces < 5K tokens; 8 rollouts with VibeThinker-1.5B as reference;
  drop problems with error rate < 0.75]`; `[+2 epochs, same hyperparams]`.
  Diversity-Exploring Distillation (probe+fuse) is applied within *both* SFT
  stages.

### 4.2 Signal phase — MGPO (`signal/mgpo`)

Per prompt `q`, sample `G` rollouts from `π_old`, score with a verifiable
reward → binary `rᵢ`. Empirical accuracy `p_c(q) = (1/G)Σ I(rᵢ=1)`.

- Max-entropy deviation distance (KL of `p_c` to `p₀ = 0.5`):
  `D_ME = p_c·log(p_c/p₀) + (1-p_c)·log((1-p_c)/(1-p₀))`.
- Weight `w_ME(p_c) = exp(-λ·D_ME)`, `λ ≥ 0`. `λ=0 ⇒ w_ME=1 ⇒ plain GRPO`
  (this identity is a required unit test).
- *Notation:* the 1.5B paper writes this coefficient `λ`; the 3B paper writes
  the identical coefficient `γ` (`w(q)=exp(-γ·D_ME)`). We use `λ` throughout.
  The 1.5B PDF also renders `D_ME` with a stray `*` between the two terms; the
  correct expression is the additive Bernoulli KL above (sum, not product),
  which the 3B paper and standard KL confirm.
- Modulated advantage `A'ⱼ(q) = w_ME(p_c(q))·Aⱼ(q)`, fed into the existing
  GRPO clipped surrogate.

**Seam (verified against mlx-go-rl source).** The `GRPOEstimator.GRPOLoss` /
`GRPOLossMasked` *methods* compute advantages internally (`grpoLossInner` →
`g.ComputeAdvantages(rewards)`), so they expose no point to inject `w_ME·A`.
The correct seam is the **package-level** `GRPOLoss(current, old, ref,
advantages, mask, config)` in `grpo.go`, which takes `advantages` as an
explicit argument and uses it directly (`ratio * advantages`). So
`signal/mgpo` = compute `w_ME` (`weight.go`); take per-group advantages from
`GroupAdvantage(rewards) [][]float64`; scale each group's entries by
`w_ME(p_c)`; materialize the `*mlx.Array` advantage tensor; call package-level
`GRPOLoss(...)`. **Scale the normalized advantage `A`, never the raw reward:**
`GroupAdvantage` normalizes by the group std, so a per-group factor on rewards
cancels — `(w·r − w·μ)/(w·σ) = (r − μ)/σ` — making reward-scaling a no-op.
`w_ME` is constant within a group, so `w_ME·A` is the only correct placement
(and matches the paper's `A'ⱼ = w_ME·Aⱼ`). Numerical guards: clamp `p_c` away
from {0,1} for the log; `D_ME ≥ 0`; `w_ME ∈ (0,1]`.

**Context schedule.** 1.5B: math RL 16K→32K then code. 3B: single 64K window
(paper found progressive truncation *hurt* the stronger 3B init), on-policy
for train/inference consistency, prefilter prompts with `p_c ∈ {0,1}` at the
phase's starting checkpoint.

**Long2Short math RL (`signal/long2short`, 3B).** A second math-RL stage
optimizing token efficiency. For each prompt group keep incorrect rewards
unchanged; for the correct set `C`, brevity `sᵢ = 1/Lᵢ`, centered shift
`r'ᵢ = rᵢ + λ·(sᵢ - s̄)/max_{j∈C}|sⱼ - s̄|`, `[λ = 0.2]` (the max is over the
correct set `C` only — incorrect traces have no brevity score). Zero-sum over `C`
(`Σ_C (r'ᵢ - rᵢ) = 0`) so the group baseline is unchanged — this invariant is
a required unit test. If all correct lengths equal, leave rewards unchanged.

**Multi-domain driver (`signal/multidomain`, 3B).** Sequential Math → Code →
STEM, each a separate MGPO run; checkpoint after each is retained for offline
self-distillation. Reward source differs per domain (answer / sandbox /
answer+option).

### 4.3 Offline Self-Distillation (`distill`, 3B)

From the retained Math/Code/STEM RL checkpoints, rejection-sample with
domain verifiers (drop incorrect). For each verified trace `y` under student
`π_stu`, learning-potential score
`S_LP(q,y) = -(1/|y|) Σ log π_stu(yₜ | q, y_<t)` (length-normalized NLL).
Higher ⇒ less well-modeled by student ⇒ more distillation value. Select within
**domain-specific length buckets** (not global), drop extremely short traces
and high-score outliers, keep the middle-to-high range, mix across domains.
Then SFT the student on this set.

### 4.4 Instruct RL (`instruct`, 3B)

On-policy RL on mixed instruction data. Reward composition:
- explicit-constraint prompts → rule-based validators (format, ordering, item
  count, keyword constraints, completion) in `reward`;
- open-ended prompts → rubric-based reward model (`reward/rubric`, gated
  interface).
Same MGPO/GRPO optimizer framework. Goal: controllability without losing
reasoning.

### 4.5 Reward sources (`reward`)

- `mathverify` — rule-based final-answer equivalence (local). 3B also pairs
  with LLM-as-judge for complex forms (gated).
- `sandbox` — execute generated code against tests; `Environment` via
  `EnvFromVerifyFunc`. Local runner with a safe-exec seam.
- `rubric` — reward-model interface, gated.
All adapt to `mlx-go-rl`'s `Environment` interface (`Score`, `Verify`).

### 4.6 Data + decontamination (`data`)

- Loaders for public datasets and the bundled parquet eval sets
  (`aime`, `aime25`, `hmmt25`, `gpqa` already present in the cloned repo).
- Synthesis/query-expansion seams (concept composition, skeletons,
  constraints) and majority-vote pseudo-labeling — *teacher-gated*.
- Quality control: n-gram repetition/degeneration filter, LLM query-quality
  filter (gated), trace-correctness filter (verifier + sandbox + majority
  vote). Stratify by trace length × difficulty for curriculum.
- `decontam`: **10-gram** overlap matching after text normalization (strip
  punctuation/symbols, unify case) to drop train samples overlapping eval.

### 4.7 Evaluation (`eval`)

- vLLM-equivalent sampling: `[temp 0.6 (code) / 1.0 (math); top_p 0.95;
  top_k -1; max 40K tokens (1.5B)]`; 3B uses `[temp 1.0, top_p 0.95,
  top_k -1, no extra length cap]`.
- Pass@1 averaged over `[k samples: math 64, code 8, knowledge 16;
  IMO-AnswerBench 16 (3B)]`, strictly binary rewards.
- **CLR (`eval/clr`, 3B):** generate `[K=32]` trajectories, extract
  `[M=5]` decision-relevant claims + final answer each; self-verify claims →
  `v_{k,m} ∈ {0,1}`; reliability `r_k = ((1/M)Σ v_{k,m})^M` (nonlinear penalty
  for any flawed claim); cluster answers by equivalence; pick the answer
  maximizing `Σ_{k∈G} r_k`; repeat the whole flow `[8×]`, report mean Pass@1.

---

## 5. Correctness strategy (what makes this a faithful repro)

Every numeric transform gets a toy-tensor unit test asserting the paper's
stated property:

1. **MGPO ≡ GRPO at λ=0** — `w_ME = 1` for all `p_c`.
2. **`w_ME` peaks at `p_c = 0.5`** and decays monotonically toward `p_c ∈
   {0,1}`; `D_ME(0.5) = 0`.
3. **Long2Short zero-sum** — `Σ_C (r'ᵢ - rᵢ) = 0`; group mean reward
   unchanged; equal-length correct set ⇒ no-op.
4. **Fusion preserves scale** — uniform merge of identical models = identity;
   weights summing to 1 preserve parameter magnitude; shape/name/dtype
   mismatch fails closed.
5. **Pass@K estimator** unbiased vs the combinatorial closed form.
6. **CLR `r_k`** = 1 iff all `M` claims valid; one invalid claim drops it
   sharply (e.g. `M=5`, 4/5 valid ⇒ `0.8^5 ≈ 0.33`).
7. **Decontam** drops a known 10-gram-overlapping sample and keeps a
   paraphrase below threshold.
8. **S_LP** ranking orders a deliberately-mispredicted trace above a
   well-modeled one.

End-to-end smoke test: a toy 2-layer model + tiny tokenizer runs the full 1.5B
stage list for a handful of steps and produces a merged + RL-updated checkpoint
without NaNs. Full benchmark reproduction stays a documented compute gate.

---

## 6. Open questions for review

1. Subdomain partition `N` and probe-set construction for **code/STEM** (paper
   only specifies math `N=4`). Inferred: define per-domain probe sets via the
   same teacher seam; surface `N` as config.
2. **Merge granularity** — whole-tensor uniform average is the paper's stated
   scheme; should we also expose per-layer weights / SLERP as opt-in, or keep
   strictly to the paper? (Lean: paper-faithful default, opt-in extension.)
3. **MGPO `p_c` source** — recomputed per optimizer step from the current
   group's rollouts (matches "on-policy"); confirm no separate probing.
4. **Reference policy / KL** — papers emphasize on-policy and a
   training/inference-consistency fix; is an explicit KL-to-reference term in
   or out for the faithful recipe? (`mlx-go-rl` supports `KLDivergence`; lean
   to off-by-default to match the on-policy emphasis, configurable.)
5. **CLR claim extraction/self-verification** are model-prompted steps — define
   as `Teacher`/self-model calls with fixed prompts, or as a separate gated
   interface?
```
