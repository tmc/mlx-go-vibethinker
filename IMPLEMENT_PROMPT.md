# Ultracode implementation prompt — mlx-go-vibethinker

Paste the block below into Claude Code (with **ultracode** on) from inside
`github.com/tmc/mlx-go-vibethinker`. It is self-contained: it points at
`DESIGN.md` as the binding contract and encodes the build order, the verified
substrate seams, the per-slice verification loop, and the hard gotchas.

---

ultracode

Fully implement the VibeThinker post-training reproduction specified in `DESIGN.md` in this repo (`github.com/tmc/mlx-go-vibethinker`). `DESIGN.md` is the binding contract — read it in full first, then implement every stage it describes, end to end, until a toy-config run of the full 1.5B and 3B recipes executes without NaNs and every algorithmic transform has a passing property test. Build like Russ Cox: minimal interfaces, errors not panics at boundaries, `doc.go` per package, table-driven tests, `gofmt`-clean, no new external deps beyond the `mlx-go` stack and stdlib.

## Ground truth and substrate
- The two papers and `DESIGN.md` are the spec. The implementation targets the *method*, not the released weights.
- Build directly on the verified on-disk substrate (real paths — confirm each symbol with `go doc`/grep before binding; do not invent APIs):
  - `github.com/tmc/mlx-go/mlx`, `.../mlx/nn`, `.../mlx/optimizer`, `.../mlx/compile`, `github.com/tmc/mlx-go/safetensors`
  - `github.com/tmc/mlx-go-lm/mlxlm` (+ `.../llm/training`, `.../llm/decode`, `.../llm/sample`, `.../llm/models` incl. Qwen2)
  - `github.com/tmc/mlx-go-examples/mlx-go-rl` (GRPO), `.../mlx-go-distill`
- These reference repos live at both `/Volumes/tmc/...` and `/Users/tmc/...` (same files). Read them; do not modify them. Wire this module to them via `go.mod` `require` + `replace` to the local paths, and keep the build green.

## Non-negotiable correctness invariants (these are where reproductions silently go wrong — verify against source, then lock with tests)
1. **MGPO scales the normalized advantage, never the raw reward.** `mlx-go-rl`'s `GroupAdvantage` normalizes by group std, so a per-group factor on rewards cancels (`(w·r−w·μ)/(w·σ)=(r−μ)/σ`) — a no-op. Compute `w_ME` per group, scale the advantage `A` (`w_ME·A`), and feed it through the **package-level** `GRPOLoss(current, old, ref, advantages, mask, config)` in `mlx-go-rl/grpo.go` (it takes `advantages` explicitly). Do **not** use `GRPOEstimator.GRPOLoss*` methods — they compute advantages internally and expose no seam. Test: `λ=0 ⇒ w_ME=1 ⇒ bit-identical to plain GRPO`; `w_ME` peaks at `p_c=0.5`, `D_ME(0.5)=0`, monotone decay toward `p_c∈{0,1}`.
2. **Long2Short reward reshape** is zero-sum over the correct set `C` only: `Σ_{C}(r'ᵢ−rᵢ)=0`, group mean unchanged, denominator `max_{j∈C}|sⱼ−s̄|`, equal-length correct set ⇒ no-op, `λ=0.2`.
3. **Expert Model Fusion** = weighted per-tensor average over safetensors, weights ≥0 summing to 1 (uniform `1/N` default). Test: uniform merge of identical models = identity; merge preserves parameter scale; name/shape/dtype mismatch fails closed.
4. **Sequence packing is NOT in `mlx-go-lm`** (only padding iterators) — build it in `spectrum/sft` as block-diagonal-mask concatenation. Don't assume it exists.
5. **Pass@K** estimator matches the unbiased combinatorial closed form when `n_samples>K`.
6. **CLR** `r_k=((1/M)Σv)^M` = 1 iff all M claims valid; one invalid claim drops it sharply (M=5, 4/5 ⇒ 0.8^5≈0.33).
7. **S_LP** (length-normalized NLL) ranks a deliberately-mispredicted trace above a well-modeled one.
8. **10-gram decontam** drops a known overlapping sample, keeps a paraphrase below threshold.
Every item in `DESIGN.md` §5 is a required test asserting the paper's stated property on toy tensors.

## Gates (implement as explicit seams, never fake)
`Teacher` (multi-path trace generation, pseudo-labels, CLR claim extraction/self-verify), `reward/rubric` reward model, `reward/sandbox` safe code execution, full-scale GPU compute, and the original corpora are pluggable interfaces with in-repo fakes for tests. A toy 2-layer model + tiny tokenizer in `internal/` must drive the whole pipeline. Mark each gate with the exact command/why it isn't run.

## Build order (dependency-first; one slice = one package or one tight pair; commit per green slice)
1. `ssp` (Stage/Pipeline/provenance types) — the orchestration spine.
2. `spectrum/fuse` (Merge) — the only genuinely-new numeric kernel; start here, it's pure and fully testable.
3. `signal/mgpo` (weight.go `D_ME`/`w_ME`, mgpo.go advantage-scaling over the GRPOLoss seam) and `signal/long2short` (reshape).
4. `reward/*` (mathverify rule-based; sandbox + rubric as gated interfaces) and `eval` (Pass@1 over k) + `eval/clr`.
5. `spectrum/probe` (passk + specialist selection) and `data` + `data/decontam`.
6. `spectrum/sft` (curriculum + sequence packing) wrapping `mlxlm` training.
7. `distill` (S_LP + length-bucket selection), `instruct`, `signal/multidomain`.
8. `cmd/vibethinker-train` (drives full 1.5B and 3B recipes from a config) and `cmd/vibethinker-eval`.
Apply the delegation seam from `DESIGN.md` §3: each exported `Foo` is a thin validate/normalize/delegate shell over an unexported `foo` core; tests target the core.

## Per-slice loop (this is the work, not a formality)
For each slice: write `doc.go` + the shell/core + property tests → `gofmt`, `go vet ./...`, `go build ./...`, `go test ./... -race` green → then verify against the papers and `mlx-go` source that the numerics match (grep the real signatures; read the actual function bodies; quote the paper formula). Use a workflow to fan out: per package, one agent implements, an independent adversarial agent tries to refute the property tests and the paper-faithfulness of the numerics (it must cite the paper passage or the `mlx-go` source line). Only land code whose invariants survive that adversarial check.

## NotebookLM cross-check (reuse the existing notebook)
A NotebookLM notebook already holds both papers, `DESIGN.md`, and the mlx-go reference bundles: notebook `c572be54-0194-4a3a-a780-2d154efc5ffc` (original Google account — if `nlm` returns exit 4 "not found", run `nlm auth` to recover the account, then `nlm account` to confirm). After each major slice, sync the new `.go` under a stable source name (`nlm source sync --include-untracked --name "repo: mlx-go-vibethinker impl" <nb> .`) and run a focused review of just that slice (faithfulness + realizability, cite sources). Treat every NotebookLM finding as a lead, not a verdict — verify it against the filesystem/papers before editing; NLM over-reports and confabulates symbols.

## Done means
- Every `DESIGN.md` stage implemented; every §5 invariant has a passing `-race` test; toy-config full 1.5B and 3B pipeline runs end to end without NaNs and emits a merged + RL-updated + distilled checkpoint with provenance.
- `gofmt`/`go vet`/`go build`/`go test ./... -race` all green; every package has a doc comment (`go list -e -f '{{.ImportPath}} :: {{.Doc}}' ./...` shows none empty).
- Committed tree carries only shippable module content (code, `doc.go`, `README.md`, `LICENSE`, tests, `cmd/`); no process residue. Use `~/bin/git-auto-commit-message --auto` for atomic commits; do not annotate commits with Claude Code; never write go.sum by hand; don't stage binaries.
- Final report: what's implemented and tested, which gates remain (with exact commands and why), and the NotebookLM verdict per slice.

Do not stop until the full toy-config pipeline runs and all invariant tests pass. If a paper detail is genuinely unspecified, implement the `DESIGN.md` §6 inferred choice behind a config knob and note it — do not block on it.
```
