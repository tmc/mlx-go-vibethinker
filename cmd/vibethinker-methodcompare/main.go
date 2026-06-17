// Command vibethinker-methodcompare runs the post-GRPO method-comparison harness
// (eval/methodcompare) and prints a comparable table plus, with -json, a
// machine-readable report.
//
// It enumerates the named configurations — the DESIGN.md baseline MGPO and each
// Tier-1/2/3 refinement from DESIGN_RL_UPGRADE.md, plus the stacked all-on — and
// reports, for a fixed seed, the mechanism metrics each knob is theorized to
// move (Dr.GRPO removes the std divisor from |A|; Clip-Higher raises the upper
// clip ceiling; Long2Short cuts tokens per sample at equal reward; DCPO-SAS
// smooths advantage variance across steps; Dynamic Sampling drops zero-gradient
// groups; HDPO activates on the cliff set; DRA reweights diversity).
//
// IMPORTANT: this is a TOY substrate. The numbers measure MECHANISM, not
// benchmark accuracy — do not read a toy delta as a paper result. The output
// header says so.
//
// Without -model the harness runs the deterministic, model-free core (the
// reproducible mechanism metrics; same seed ⇒ identical numbers). With -model
// (which requires the modelir build tag) it additionally runs each method's loss
// through a toy Qwen2 model and reports the model-coupled, NON-reproducible
// final-loss and wall-time.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
)

func main() {
	seed := flag.Uint64("seed", 1, "scenario/model seed (same seed ⇒ identical core metrics)")
	asJSON := flag.Bool("json", false, "emit the machine-readable JSON report instead of the table")
	withModel := flag.Bool("model", false, "also run each method's loss through the toy model (requires -tags modelir); adds non-reproducible loss/wall-time")
	flag.Parse()

	if err := run(*seed, *asJSON, *withModel); err != nil {
		fmt.Fprintf(os.Stderr, "vibethinker-methodcompare: %v\n", err)
		os.Exit(1)
	}
}

func run(seed uint64, asJSON, withModel bool) error {
	var (
		metrics []methodcompare.Metrics
		err     error
	)
	if withModel {
		metrics, err = evaluateWithModel(seed)
	} else {
		metrics, err = methodcompare.Evaluate(seed)
	}
	if err != nil {
		return err
	}

	if asJSON {
		doc, err := methodcompare.NewReport(seed, metrics).JSON()
		if err != nil {
			return err
		}
		_, err = os.Stdout.Write(doc)
		return err
	}

	if withModel {
		fmt.Print(methodcompare.TableWithModel(seed, metrics))
	} else {
		fmt.Print(methodcompare.Table(seed, metrics))
	}
	return nil
}
