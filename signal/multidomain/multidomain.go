package multidomain

import (
	"context"
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// A Domain is one stage of the multi-domain RL sequence: a name and the reward
// source used to score that domain's rollouts.
type Domain struct {
	Name   string
	Reward rl.Environment
}

// DefaultOrder returns the paper's domain order: Math, then Code, then STEM.
// The caller supplies the reward source for each.
func DefaultOrder(math, code, stem rl.Environment) []Domain {
	return []Domain{
		{Name: "math", Reward: math},
		{Name: "code", Reward: code},
		{Name: "stem", Reward: stem},
	}
}

// A DomainRunner performs one domain's MGPO run, starting from inDir and using
// the domain's reward source, and returns the output checkpoint directory. It
// is the seam by which the MGPO optimizer plugs in; tests inject a fake.
type DomainRunner interface {
	RunDomain(ctx context.Context, d Domain, inDir string) (outDir string, err error)
}

// A Result records the checkpoint retained after a domain's RL run, in order.
// All checkpoints are retained because Offline Self-Distillation samples from
// each.
type Result struct {
	Domain string
	Dir    string
}

// Run executes the domains in order, threading the checkpoint from one to the
// next and retaining every intermediate checkpoint. It returns the per-domain
// results (in order) and the final checkpoint directory. A domain with a nil
// reward source is an error.
func Run(ctx context.Context, runner DomainRunner, inDir string, domains []Domain) ([]Result, string, error) {
	if runner == nil {
		return nil, "", fmt.Errorf("multidomain: nil runner")
	}
	cur := inDir
	results := make([]Result, 0, len(domains))
	for i, d := range domains {
		if err := ctx.Err(); err != nil {
			return nil, "", fmt.Errorf("multidomain: canceled before domain %q: %w", d.Name, err)
		}
		if d.Reward == nil {
			return nil, "", fmt.Errorf("multidomain: domain %d %q has nil reward", i, d.Name)
		}
		out, err := runner.RunDomain(ctx, d, cur)
		if err != nil {
			return nil, "", fmt.Errorf("multidomain: domain %d %q: %w", i, d.Name, err)
		}
		results = append(results, Result{Domain: d.Name, Dir: out})
		cur = out
	}
	return results, cur, nil
}
