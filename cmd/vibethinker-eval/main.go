// Command vibethinker-eval runs VibeThinker's evaluation harness: Pass@1 over k
// samples and CLR (Claim-Level Reliability) test-time scaling.
//
// Generation from a real model is a gate (it needs a trained checkpoint and
// decode loop), so this command drives the real estimators with in-repo fakes:
// a fixed-answer sampler scored by the rule-based math verifier for Pass@1, and
// the deterministic CLR fake verifier for CLR. The arithmetic exercised — the
// Pass@1 mean-of-means and the CLR reliability r_k = ((1/M)Σv)^M and
// reliability-weighted answer selection — is the same a real run would use.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/tmc/mlx-go-vibethinker/eval"
	"github.com/tmc/mlx-go-vibethinker/eval/clr"
	"github.com/tmc/mlx-go-vibethinker/reward/mathverify"
)

func main() {
	mode := flag.String("mode", "pass1", "evaluation mode: pass1 or clr")
	flag.Parse()
	if err := run(*mode); err != nil {
		fmt.Fprintf(os.Stderr, "vibethinker-eval: %v\n", err)
		os.Exit(1)
	}
}

func run(mode string) error {
	switch mode {
	case "pass1":
		return runPass1()
	case "clr":
		return runCLR()
	default:
		return fmt.Errorf("unknown mode %q (want pass1 or clr)", mode)
	}
}

// runPass1 evaluates Pass@1 with a fixed-answer sampler (the model-generation
// gate stands in as a fake) scored by the rule-based math verifier.
func runPass1() error {
	ctx := context.Background()
	// The "model" always answers 4; the verifier checks against each gold.
	sampler := eval.SamplerFunc(func(ctx context.Context, prompt string, p eval.Params, n int) ([]string, error) {
		out := make([]string, n)
		for i := range out {
			out[i] = `the answer is \boxed{4}`
		}
		return out, nil
	})
	prompts := []string{"2+2", "1+3", "9-1"} // golds 4, 4, 8
	golds := []string{"4", "4", "8"}
	// Score each prompt against its own gold via a combined verifier.
	var total float64
	params := eval.MathParams
	for i, prompt := range prompts {
		res, err := eval.Pass1(ctx, sampler, mathverify.Environment(golds[i]), []string{prompt}, params, 0.5)
		if err != nil {
			return err
		}
		total += res.PassAt1
		fmt.Printf("prompt %q gold %q: pass@1 = %.3f\n", prompt, golds[i], res.PassAt1)
	}
	fmt.Printf("mean pass@1 = %.3f (math sampler answers 4; correct on 2 of 3)\n", total/float64(len(prompts)))
	return nil
}

// runCLR evaluates CLR with the deterministic fake verifier, showing the
// reliability-weighted selection.
func runCLR() error {
	ctx := context.Background()
	// Two answer clusters: "4" backed by fully-reliable trajectories, "5" by a
	// flawed one. CLR should pick "4".
	fv := clr.FakeVerifier{Trajs: []clr.Trajectory{
		{Answer: "4", Claims: []int{1, 1, 1, 1, 1}},
		{Answer: "4", Claims: []int{1, 1, 1, 1, 1}},
		{Answer: "5", Claims: []int{1, 1, 1, 1, 0}},
	}}
	eq := func(a, b string) bool { return a == b }
	res, err := clr.Score(ctx, fv, "2+2", "4", 6, 5, 4, eq, eq)
	if err != nil {
		return err
	}
	selected := ""
	if len(res.Answers) > 0 {
		selected = res.Answers[0]
	}
	fmt.Printf("CLR selected answer = %q (gold 4); mean pass@1 = %.3f\n", selected, res.MeanPass)
	return nil
}
