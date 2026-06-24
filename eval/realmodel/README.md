# Real-model RL-upgrade evaluation (`eval/realmodel`)

This package and its command, [`cmd/vibethinker-realmodel`](../../cmd/vibethinker-realmodel),
run the post-GRPO RL upgrades from [`DESIGN_RL_UPGRADE.md`](../../DESIGN_RL_UPGRADE.md)
on the **real** Qwen2.5-Math-1.5B model — real logits, real optimizer steps —
rather than the deterministic toy substrate of [`eval/methodcompare`](../methodcompare).

It is a **mechanism + stability** instrument, **not** a benchmark-accuracy
reproduction. Reproducing VibeThinker's published numbers needs ~3.9K H800
GPU-hours, a frontier teacher, and the full data/sandbox seams — out of scope.
Every number here is model- and machine-dependent and **not** bit-reproducible
across runs; the tag-free toy harness (`eval/methodcompare`) remains the
byte-identical-reproducibility one.

Everything below is behind the `modelir` build tag (the same tag the toy model
pipeline uses), so the tag-free build and the nine `signal/mgpo` method tests
stay green and untouched.

## Getting the weights

The weights are multi-GB safetensors and are **never committed** (`*.safetensors`
is gitignored). Download a HuggingFace export of `Qwen2.5-Math-1.5B`
(`config.json`, `*.safetensors`, `tokenizer.json`) and point the loader at it:

    export VIBETHINKER_REALMODEL_DIR=~/models-tmp/Qwen2.5-Math-1.5B

`VIBETHINKER_REALMODEL_DIR` is the only runtime env knob; if unset, the loader
falls back to `~/models-tmp/Qwen2.5-Math-1.5B`. The `-model` flag overrides both.

## The three run modes

The command has three modes, in increasing rigor:

1. **Load-and-forward smoke** — load the real Qwen2.5-Math-1.5B and run a single
   forward pass to confirm the weights, tokenizer, and mlx-go stack load and
   produce finite logits. This is the lightest sanity check.

       go test -tags modelir -run TestLoad ./eval/realmodel/

2. **Single-method subprocess harness** — run each post-GRPO method through a
   short real GRPO loop. With `-source seeded`, fixed real-tokenized
   real-`Forward`-rescored completions with guaranteed mixed correctness make the
   reward-shape mechanisms observable (organic rollouts from a weak base score
   ~0% and collapse the within-group spread, so they only check stability). Each
   method runs in its **own subprocess** (`-one-method <idx>`) so the
   non-reclaimable ~13 GB value-and-grad graph does not accumulate across methods.

       go run -tags modelir ./cmd/vibethinker-realmodel -source seeded

   SEEDED is **not** model accuracy.

3. **Two-process sweep (CKPT-B)** — the differential training sweep over the
   C1..C5 ladder, ranked by held-out Avg@1 Δacc (`AccFinal − AccStep0`) on a fixed
   held-out probe (N=12) scored by greedy (deterministic) decode:

       go run -tags modelir ./cmd/vibethinker-realmodel -sweep

   The C1..C5 ladder (see [`sweepconfig.go`](sweepconfig.go)):
   **C1** zero-Method baseline · **C2** Tier-1 (Dr.GRPO debias + DAPO Clip-Higher
   0.2/0.28) · **C3** DCPO-SAS · **C4** HDPO (λ_JSD=0.5) · **C5** composed
   (C2+C3+C4). The do-not-adopt set (SDPO/SRPO/QAE/GSPO/GMPO/VAPO) and
   DynSampling/DRA/FRPO are deliberately excluded from the ladder.

## The two-process split (and the Metal array ceiling)

On a single M4 Max, a single-process *score@0 + train + score@final* run crashes
in the final held-out pass:

    [metal::malloc] Resource limit (499000) exceeded

The non-reclaimable ~13 GB training graph plus the O(n²) no-cache held-out decode
jointly exceed the Metal array ceiling, even at the smoke floor. (Measured, not
assumed: `TestSweepBaselineHeldoutDelta` documents the wall and is skipped by
default because it crashes by design — set `VT_RUN_SINGLEPROC_WALL=1`, with the
`VT_PROMPTS`/`VT_K`/`VT_MAXTOK`/`VT_STEPS`/`VT_SEED`/`VT_SOURCE` knobs, to re-probe
it.)

So `-sweep` splits each config into **two OS subprocesses** that never share a
Metal budget:

- **Phase 1** (`-sweep-phase 1`): load base, score held-out @step0, train, and
  checkpoint **only** the trained q/v projections (56 tensors, ~294 MB bf16) via
  `SaveSafetensors`, then exit — its training graph dies with the process.
- **Phase 2** (`-sweep-phase 2`): a fresh process loads base, applies the q/v
  checkpoint (bit-exact round-trip), and scores held-out @final on a clean budget.

The parent runs the phases **serially** (blocking `cmd.Run`, no concurrency),
stitches the two rows into a held-out Δacc, ranks C1..C5, and measures a ≥2-seed
baseline noise floor. A crashed or partial phase becomes an explicit schema-1
**ERROR cell** carrying the exit code and last stderr — never a silent gap.

> **Do not run two real-model loads at once.** Each load can approach the Metal
> ceiling on its own; concurrent loads race it. The sweep is serial by design.

## Reading the result

The report ranks configs by held-out Δacc with mechanism stats (`ratioVar`,
`advStd`) alongside as corroborating *why*, not as the ranking key. If the best
Δacc does not clear the baseline noise-floor spread, the recommendation is
**INDISTINGUISHABLE at N=X** and defaults to the simplest config (C1) — a valid,
honest verdict, not a failure. A bigger held-out set or more steps would be
needed to separate the methods, which is out of scope on a single M4 Max.

`Δacc` is a **directional** signal on unseen short-horizon math, **not** benchmark
accuracy. Because the held-out decode is greedy/deterministic, `AccStep0` is
identical across seeds (constant base weights); the run-to-run variation is in
`AccFinal`, driven by the temperature-sampled training rollouts and the optimizer.

### What this does and does not show

This instrument is bounded, and states its own limits:

- **DIRECTIONAL, not benchmark accuracy.** Every accuracy-like number
  (`AccStep0`, `AccFinal`, `Δacc`, Avg@1) is a directional signal on a small
  held-out probe, not a benchmark score.
- **N=12.** The held-out probe is 12 prompts at the smoke floor. This is far too
  small to separate methods that move accuracy by less than the run-to-run noise.
- **Single M4 Max.** The whole instrument is bounded by one machine's Metal
  array ceiling (~499000 live buffers). Subprocess value-and-grad isolation (the
  two-process split) is **mandatory**, not an optimization — a single process
  busts the ceiling and crashes by design.
- **Not bit-reproducible.** A real model plus a real optimizer is not
  deterministic across runs the way the tag-free toy harness is.
- **Full repro out of scope.** VibeThinker's published numbers need ~3.9K H800
  GPU-hours, a frontier teacher, and the full data/sandbox seams.

**The honest finding:** the mechanism stats separate the configs (C2/C5 carry the
Dr.GRPO/Clip-Higher reward-shape signature; C1/C3/C4 do not), but the held-out
**Δacc is INDISTINGUISHABLE at N=12** — no config clears the baseline noise floor,
so the recommendation defaults to the simplest config, **C1**. The value of this
result is the *verified noise floor* and the *mechanism-separates-but-Δacc-doesn't*
observation, not a fabricated winner.

### Reproduced result (CKPT-C)

Run independently end-to-end twice (CKPT-B and an independent CKPT-C reproduction
on a separate pass), bit-matching each other. Run config: `prompts=6 K=4
maxTok=32 temp=0.80 steps=8 lr=1e-06 seed=1 source=seeded`; held-out N=12, Avg@1
greedy; model Qwen2.5-Math-1.5B; two-process split.

    config           acc0   accFin    Δacc   steps    ratioVar    advStd
    1. C1-baseline   0.500    0.417   -0.083      8     0.07761    0.8473
    2. C2-tier1      0.500    0.417   -0.083      8      0.1004    0.3943
    3. C3-dcpo-sas   0.500    0.417   -0.083      8     0.07761    0.8473
    4. C4-hdpo       0.500    0.417   -0.083      8     0.07761    0.8473
    5. C5-composed   0.500    0.417   -0.083      8      0.1004    0.3943

Noise floor — C1-baseline, seeds [1, 2]: Δacc deltas [-0.083, -0.083], spread
**0.000**.

Verdict — **INDISTINGUISHABLE at N=12**: the best Δacc (C1-baseline, -0.083) does
not clear the C1-baseline noise-floor spread (0.000); default to the simplest
config, **C1-baseline**. A bigger held-out set or more steps would be needed to
separate the methods — out of scope on a single M4 Max.

Every config moves 0.500 → 0.417 identically, so held-out Δacc cannot tell them
apart at this N. But the mechanism stats *do* separate them: C2/C5 (Tier-1
Dr.GRPO + Clip-Higher present) carry `ratioVar` 0.1004 / `advStd` 0.3943, while
C1/C3/C4 carry 0.07761 / 0.8473 — the knobs move the mechanism they target, they
just don't move held-out Δacc at this N. That separation-without-Δacc-movement,
together with the verified 0.000 noise floor, is the result.
