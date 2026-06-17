# DESIGN: Post-GRPO RL upgrades for the VibeThinker pipeline

Status: proposal (no code yet). Companion to [`DESIGN.md`](./DESIGN.md), which is
the binding contract for the baseline reproduction. This document proposes how to
make the **Signal** (RL) phase train *more effectively* by adopting verified
better-than-GRPO methods from 2025–2026, ranked for the **1.5B/3B** scale this
project actually targets.

The baseline RL is MGPO (GRPO + a max-entropy advantage weight `w_ME`), Long2Short
length shaping, and multi-domain sequential RL. The seam is the package-level
`rl.GRPOLoss(current, old, ref, advantages, mask, config)` in `mlx-go-rl/grpo.go`;
MGPO multiplies `w_ME` onto the **normalized** group-relative advantage before that
call (`signal/mgpo`).

## 0. The constraint that decides everything: small-model viability

VibeThinker is 1.5B/3B. The single biggest finding of this research is that **the
most-hyped post-GRPO method, SDPO (Self-Distillation Policy Optimization), is
actively harmful at our scale.** SDPO's own paper (arXiv 2601.20802, §4.1 / Fig. 17)
states verbatim that *"SDPO underperforms GRPO on Qwen2.5-1.5B"*; its gains
(4–10× faster, +7–30 pts) are an emergent property of ≥8B models with strong
in-context learning. Its successor SRPO (arXiv 2604.02288) repairs SDPO's collapse
by routing only failed rollouts to self-distillation, but its smallest tested model
is **Qwen3-4B** — it is *not validated at ≤3B*.

So the ranking below is governed by a rule: **prefer pure objective/advantage/
sampling changes whose gains are validated at ≤3B (or are mechanically scale-
agnostic), and which compose with `w_ME` and Long2Short. Treat self-distillation as
gated/experimental, not core.**

Every method here was read from the primary paper and adversarially verified
against its experiments table; sources are listed in §6. Per project practice,
NotebookLM/LLM literature summaries were treated as leads and confirmed against
source (one title and one "7B-only" claim were confabulations, corrected here).

## 1. Where each method touches the code (verified against `grpo.go`)

The existing seam hardcodes a **per-token** ratio/clip/min surrogate and exposes
`advantages`, `ClipEpsLow/High`, and a `DrGRPO` toggle via `GRPOConfig`. That
determines integration cost:

| Layer | Methods that land here | Cost |
| --- | --- | --- |
| **Advantage value** (what MGPO already edits) | Dr.GRPO std-removal, DCPO SAS standardization, DRA-GRPO reward reweight, QAE baseline, FRPO future-KL term | low — same seam as `w_ME` |
| **`GRPOConfig` flags** (already present) | DAPO Clip-Higher (`ClipEpsLow/High`), Dr.GRPO length-norm (`DrGRPO`) | ~free |
| **Loss body** of `GRPOLoss` / new sibling loss | DAPO token-level aggregation, GSPO sequence-level ratio | medium — edit/clone the surrogate |
| **Rollout / data layer** (upstream of loss) | DAPO Dynamic Sampling, DCPO cumulative per-prompt stats, overlong reward shaping | medium — new state/plumbing |
| **Gated self-distill branch** (new) | SDPO dense logit advantage, SRPO routing, HDPO cliff-JSD | high — extra forward pass + branch |

## 2. Recommended upgrades, ranked (the actual proposal)

### Tier 1 — adopt now: free, scale-safe, compose cleanly with MGPO

**(1) Dr.GRPO advantage/loss debiasing** *(arXiv 2503.20783)*
Remove GRPO's two biases: the per-response length divisor `1/|o_i|` (replace with a
global constant `L`) and the question-level std divisor in the advantage
(`A_i = r_i − mean(r)` instead of `(r_i − mean)/std`). Pure algebraic change; the
substrate **already has** `GroupAdvantageDrGRPO` (no std) and a `DrGRPO` config
toggle for the length branch.
- *Why:* removes the systematic length inflation (esp. of *wrong* answers) and the
  difficulty bias that make GRPO waste tokens; preserves/slightly improves accuracy.
- *Composition:* `w_ME` still multiplies the (now un-normalized) advantage — verify
  the `w_ME` magnitude is still sensible without the `/std` (it was tuned against a
  std-normalized `A`; re-check λ). **Synergistic with Long2Short** (both fight length
  bloat) but they must not double-count: Long2Short shapes the *reward* within the
  correct set; Dr.GRPO changes the *normalization*. Keep them orthogonal.
- *Scale:* shown at 7B; mechanically scale-agnostic. **Low risk.**

**(2) DAPO Clip-Higher** *(arXiv 2503.14476)*
Decouple the PPO clip range: `clip(r, 1−ε_low, 1+ε_high)` with `ε_low=0.2,
ε_high=0.28` instead of a symmetric `ε`. The config fields **already exist**
(`ClipEpsLow/High`) — this is a hyperparameter change plus a short ablation.
- *Why:* the symmetric upper clip caps the probability growth of low-probability
  "exploration" tokens and drives entropy collapse; raising the ceiling preserves
  exploration diversity. Directly protects the *diversity* that VibeThinker's whole
  Spectrum phase exists to build.
- *Composition:* independent of `w_ME` and Long2Short. **Low risk.**

### Tier 2 — adopt with modest plumbing: validated at ≤3B

**(3) DCPO — Dynamic clipping + Smooth Advantage Standardization** *(arXiv 2509.02333)*
**Verified evaluated on Qwen2.5-Math-1.5B and Qwen2.5-3B** (no extra model). Three
parts: Dynamic-Adaptive Clipping bounds as a closed form of the old-policy prob;
Smooth Advantage Standardization (SAS) that standardizes the reward over a
*cumulative* per-prompt history and picks the min-magnitude of two smoothed
estimates; Only-Token-Mean loss aggregation.
- SAS (verbatim): `SÂ_new = ((i−1)/i)·Â_new + (1/i)·Â_total`,
  `SÂ_total = (1/i)·Â_new + ((i−1)/i)·Â_total`, final
  `Â = SÂ_new if |SÂ_new| < |SÂ_total| else SÂ_total`.
- *Why:* lifts "response utilization" (fraction of rollouts with nonzero gradient)
  ~43.8%→71.8% and roughly halves training time vs DAPO — i.e. **sample/compute
  efficiency**, the main thing we can demonstrate at small scale.
- *Cost:* SAS requires carrying per-prompt running reward mean/std **across steps**
  (cumulative state) — the one nontrivial addition; lands at the advantage seam, so
  `w_ME` still multiplies the result. **Caveat:** at ≤3B the *accuracy* gain over
  GRPO is modest (+1.2/+1.3); the value is efficiency, not a headline accuracy jump.
- *Decision:* DCPO's DAC and Dr.GRPO/DAPO clip changes are alternative clip schemes —
  **pick one clip strategy**, don't stack DAC on Clip-Higher blindly. Recommend
  prototyping DCPO-SAS (the efficiency lever) on top of Tier-1 Dr.GRPO+Clip-Higher,
  and A/B DAC vs Clip-Higher.

### Tier 3 — optional / experimental: real but small or costly at our scale

**(4) FRPO — Future-KL regularization** *(arXiv 2601.10201)* — *cleanest pure
drop-in.* Names GRPO explicitly; needs no critic. The fix is nearly one line:
*"add a reverse cumulative sum of per-token log ratios after advantage
construction"* (a causal "future-regularization return-to-go" the local token-KL
omits). Reported: higher pass@16, higher entropy, lower policy drift — *in a
large-model setting*; no ≤3B result. Lands at the advantage seam, composes with
`w_ME`. **Adopt if late-stage policy drift / entropy collapse is observed on long
3B traces; otherwise hold.**

**(5) HDPO — privileged self-distillation on "cliff" groups** *(arXiv 2603.23871)*
**Verified at Qwen2.5-Math-1.5B, no external model** (self-teacher = same weights
conditioned on the gold answer `y*`). On zero-reward groups (all rollouts fail, GRPO
gradient vanishes), generate privileged rollouts `ȳ ~ π_θ(·|x⊕y*)`, keep correct
ones, add `L_HDPO = L_GRPO + λ·L_JSD` where `L_JSD` distills the policy toward its
own privileged distribution (top-k=64). *Why:* fixes the hard-question gradient
cliff (improves pass@k coverage) — directly relevant to VibeThinker's hardest
curriculum buckets. **But gains are sub-2%, near run-to-run noise.** Treat as a
low-cost exploration aid for the hard-data RL stage, gated behind the existing
verifiable-reward seam. This is the **small-model-safe form of self-distillation** —
prefer it over SDPO/SRPO at our scale.

**(6) DRA-GRPO — diversity-aware reward reweight** *(arXiv 2505.09655)* — validated
at 1.5B (DeepSeek-R1-Distill-Qwen-1.5B, 58.2% avg, ~$55 run). Reweights each
rollout's reward by inverse submodular-mutual-information to its group siblings
(`R_i ← R_i / Σ_j sim(o_i,o_j)`, O(G²) over a cosine matrix) before advantage
normalization. *Why:* fights mode collapse / diversity–quality inconsistency — again
aligned with the Spectrum goal. **Hard cost: an external sentence embedder**
(`jina-embeddings-v2-small-en`), which violates the "no new model / self-contained"
principle. **Only adopt if mode collapse is empirically observed and the embedder is
acceptable as a gated dependency.**

### Do NOT adopt (for the 1.5B/3B target)

- **SDPO** (2601.20802): verbatim *underperforms* GRPO at 1.5B. ✗
- **SRPO** (2604.02288): promising routing idea but smallest tested model is 4B; EMA
  self-teacher + feedback channel is real complexity for unproven ≤3B benefit. Revisit
  if scaling above 4B. If self-distillation is wanted now, use HDPO instead.
- **QAE** (2509.22611): clean baseline swap (mean→K-quantile) but validated **only at
  ≥8B**; not reported small. Hold.
- **GSPO / GMPO** (2507.18071 / 2507.xxxxx): sequence-level / geometric-mean ratios,
  validated on 7B–30B (MoE). Require a **new sibling loss** (`GSPOLoss`) — the
  per-sequence ratio `exp((1/|y|)Σ log-ratio)` is not reproducible via the
  `advantages` argument. Defer unless training instability on long 3B sequences
  specifically points at token-level ratio variance.
- **VAPO** (2504.05118): needs a learned **value critic** — conflicts with the
  critic-free GRPO substrate. ✗

## 3. Proposed phasing

1. **Phase A (free wins, validate the harness):** land Dr.GRPO debiasing + DAPO
   Clip-Higher behind `GRPOConfig` flags. Re-tune MGPO `λ` against the now
   un-std-normalized advantage. Property tests: (a) `DrGRPO` off ⇒ bit-identical to
   today; (b) Clip-Higher with `ε_low=ε_high` ⇒ identical to symmetric clip; (c)
   `w_ME` composition unchanged (λ=0 ⇒ plain Dr.GRPO advantage).
2. **Phase B (efficiency):** add DCPO-SAS (cumulative per-prompt reward stats) +
   DAPO Dynamic Sampling (drop groups with accuracy 0 or 1 — note this *overlaps* the
   existing `std=0` zero-advantage guard, so unify them). Measure response-
   utilization and steps-to-target, not just final accuracy.
3. **Phase C (gated, hard-data only):** prototype HDPO cliff-JSD on the hardest
   curriculum bucket as an exploration aid; keep FRPO and DRA-GRPO as flags to switch
   on only if drift / mode-collapse is observed.

## 4. Invariants the upgrade must preserve (extend `DESIGN.md` §5)

- **MGPO no-op rule still holds:** every advantage edit (Dr.GRPO, DCPO-SAS, DRA
  reweight, FRPO) happens *before* `w_ME` multiplication and *before* `GRPOLoss`, on
  the normalized (or deliberately un-normalized) advantage — never on raw rewards
  inside a std-normalized path (that cancels; see `DESIGN.md` §4.2).
- **Long2Short zero-sum** over the correct set is unchanged; Dr.GRPO's std-removal
  must not silently rescale the Long2Short shift — test that Long2Short Δ over `C`
  stays 0 after the normalization change.
- **Every new method is a config-gated branch with an off-path identical to today**,
  so the baseline reproduction in `DESIGN.md` is never disturbed.
- **No new learned component** for Tier 1–2 and HDPO/FRPO (self-contained); DRA's
  embedder is the *only* sanctioned external model and must be a gated seam with an
  in-repo fake, exactly like Teacher/rubric.

## 5. Open questions

- Does Dr.GRPO's std-removal destabilize MGPO's `w_ME` (which was derived assuming a
  std-normalized advantage)? Needs an ablation; may require re-deriving the `w_ME`
  scale.
- DCPO DAC vs DAPO Clip-Higher: are they redundant or complementary at ≤3B? A/B.
- Is the gradient-accumulation correctness issue flagged by arXiv 2604.23747 (the
  DeepSpeed/OpenRLHF micro-batch + loss-aggregation bugs that *silently suppress
  SFT*) reproducible in the mlx-go training loop? Worth a guard test — it validates
  VibeThinker's sequential SFT-then-RL design but warns the baseline can be quietly
  wrong.

## 6. Verified sources

- SDPO — "Reinforcement Learning via Self-Distillation", arXiv 2601.20802 (small-model
  hazard: §4.1 / Fig. 17, *verbatim* underperforms GRPO at 1.5B).
- SRPO — "Unifying Group-Relative and Self-Distillation Policy Optimization via Sample
  Routing", arXiv 2604.02288 (smallest tested 4B).
- DAPO — "An Open-Source LLM RL System at Scale", arXiv 2503.14476 (Clip-Higher,
  Dynamic Sampling, token-level loss, overlong shaping; 32B).
- Dr.GRPO — "Understanding R1-Zero-Like Training", arXiv 2503.20783 (length + std
  debiasing).
- DCPO — "Dynamic Clipping Policy Optimization", arXiv 2509.02333 (**validated 1.5B +
  3B**; DAC + SAS + OTM).
- HDPO — "Hybrid Distillation Policy Optimization via Privileged Self-Distillation",
  arXiv 2603.23871 (**validated 1.5B**; cliff-JSD, self-teacher, no external model).
- DRA-GRPO — "Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training",
  arXiv 2505.09655 (validated 1.5B; **needs jina embedder**).
- FRPO — "Future-KL Regularized GRPO", arXiv 2601.10201 (one-line future-KL; names
  GRPO; large-model only).
- QAE — "Quantile Advantage Estimation", arXiv 2509.22611 (≥8B only).
- GSPO — "Group Sequence Policy Optimization", arXiv 2507.18071 (7B–30B; sequence-level
  ratio).
- VAPO — arXiv 2504.05118 (value critic; conflicts with critic-free design).
- SFT-then-RL > mixed-policy — arXiv 2604.23747 (validates sequential design; warns of
  silent SFT-suppressing framework bugs).
- Stage-specific small-model data — arXiv 2606.04466 (difficulty apportionment; ≤1B
  tested).

A NotebookLM notebook (`c572be54-0194-4a3a-a780-2d154efc5ffc`) holds the full text of
these papers plus the VibeThinker papers, `DESIGN.md`, and the mlx-go reference
sources for further cross-checking.
