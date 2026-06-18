// Command vibethinker-realmodel runs the real-model mechanism smoke test
// (eval/realmodel): it loads the real Qwen2.5-Math-1.5B, runs each post-GRPO
// method (DESIGN_RL_UPGRADE.md) through a short real GRPO loop on real logits,
// and prints a comparison table plus, with -json, a machine-readable report.
//
// IMPORTANT: this validates each method's MECHANISM and training STABILITY on
// real logits — NOT benchmark accuracy. Reproducing VibeThinker's published
// numbers needs ~3.9K H800 GPU-hours and is out of scope. Every number is model-
// and machine-dependent and NOT reproducible across runs; the tag-free toy
// harness (eval/methodcompare) remains the byte-identical-repro one.
//
// The report has two clearly-labeled blocks: ORGANIC (honest model-generated
// rollouts — a weak base scores ~0%, collapsing within-group reward spread) and
// SEEDED (fixed, real-tokenized, real-Forward-rescored completions with
// guaranteed mixed correctness, so the reward-shape mechanisms are observable;
// SEEDED is NOT model accuracy).
//
// This command requires the modelir build tag (it loads the real model):
//
//	go run -tags modelir ./cmd/vibethinker-realmodel
//
// Point it at the model dir with -model or VIBETHINKER_REALMODEL_DIR.
package main

import (
	"flag"
	"fmt"
	"os"
)

// opts bundles the CLI configuration shared by the parent orchestrator and the
// single-method child.
type opts struct {
	model     string
	asJSON    bool
	prompts   int
	k         int
	maxTokens int
	steps     int
	seed      uint64

	// Child mode: when oneMethod >= 0 this process runs exactly ONE method
	// (registry index) under source and prints a single JSON ChildRow, then
	// exits — the subprocess-per-method isolation that keeps the non-reclaimable
	// ~13GB value-and-grad graph from accumulating across methods.
	oneMethod int
	source    string
}

func main() {
	var o opts
	flag.StringVar(&o.model, "model", "", "model directory (default: $VIBETHINKER_REALMODEL_DIR or ~/models-tmp/Qwen2.5-Math-1.5B)")
	flag.BoolVar(&o.asJSON, "json", false, "emit the machine-readable JSON report instead of the table")
	flag.IntVar(&o.prompts, "prompts", 6, "number of math prompt groups")
	flag.IntVar(&o.k, "k", 4, "rollouts per prompt (>=4 for real within-group spread)")
	flag.IntVar(&o.maxTokens, "max-tokens", 32, "max generated tokens per rollout (bounded for the Metal array ceiling)")
	flag.IntVar(&o.steps, "steps", 8, "real optimizer steps per method")
	flag.Uint64Var(&o.seed, "seed", 1, "rollout RNG seed")
	flag.IntVar(&o.oneMethod, "one-method", -1, "child mode: run only this registry method index and print one JSON row (internal)")
	flag.StringVar(&o.source, "source", "", "child mode: rollout source for -one-method (organic|seeded) (internal)")
	flag.Parse()

	if err := run(o); err != nil {
		fmt.Fprintf(os.Stderr, "vibethinker-realmodel: %v\n", err)
		os.Exit(1)
	}
}
