// Command vibethinker-train drives a VibeThinker Spectrum-to-Signal training
// run from a config.
//
// It assembles the recipe for the requested model size (1.5b or 3b) as an ssp
// pipeline and runs it, printing per-stage progress and the final checkpoint's
// provenance. With -toy (the default), it runs the toy Qwen2 pipeline that
// exercises every algorithmic transform on CPU without GPU-scale compute; a real
// run supplies a base model path and the external gates (teacher, corpora,
// compute) described in DESIGN.md.
//
// The toy pipeline requires the model registry, which is built under the modelir
// tag; build with -tags modelir.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/tmc/mlx-go-vibethinker/ssp"
)

func main() {
	size := flag.String("size", "1.5b", "recipe size: 1.5b or 3b")
	toy := flag.Bool("toy", true, "run the toy CPU pipeline (the only mode without GPU-scale compute and external gates)")
	out := flag.String("out", "", "working directory for checkpoints (default: a temp dir)")
	seed := flag.Uint64("seed", 1, "toy model seed")
	flag.Parse()

	if err := run(*size, *toy, *out, *seed); err != nil {
		fmt.Fprintf(os.Stderr, "vibethinker-train: %v\n", err)
		os.Exit(1)
	}
}

func run(size string, toy bool, out string, seed uint64) error {
	if !toy {
		return fmt.Errorf("non-toy runs require a base model and the external gates (teacher, corpora, GPU compute) from DESIGN.md §1; only -toy is wired here")
	}
	dir := out
	if dir == "" {
		d, err := os.MkdirTemp("", "vibethinker-train-")
		if err != nil {
			return err
		}
		dir = d
	}
	pipe, err := buildToyPipeline(size, dir, seed)
	if err != nil {
		return err
	}
	pipe.Observe = func(ev ssp.Event) {
		switch ev.Kind {
		case ssp.StageStart:
			fmt.Printf("[%d] %s: running\n", ev.Index, ev.Stage)
		case ssp.StageDone:
			note := ""
			if len(ev.Out.History) > 0 {
				note = ev.Out.History[len(ev.Out.History)-1].Note
			}
			fmt.Printf("[%d] %s: done — %s\n", ev.Index, ev.Stage, note)
		case ssp.StageError:
			fmt.Printf("[%d] %s: ERROR %v\n", ev.Index, ev.Stage, ev.Err)
		}
	}
	final, err := pipe.Run(context.Background(), &ssp.Checkpoint{})
	if err != nil {
		return err
	}
	fmt.Printf("\nfinal checkpoint: %s\n", final.Dir)
	fmt.Printf("provenance (%d stages):\n", len(final.History))
	for _, p := range final.History {
		fmt.Printf("  %-12s %s\n", p.Stage, p.Note)
	}
	return nil
}
