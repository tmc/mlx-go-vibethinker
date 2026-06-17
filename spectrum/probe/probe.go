package probe

import (
	"context"
	"fmt"
)

// MathSubdomains is the paper's math partition into N = 4 subdomains. Other
// domains (code, STEM) surface their own subdomain list as config (DESIGN §6.1);
// the partition is data, not hard-coded into the selector.
var MathSubdomains = []string{"algebra", "geometry", "calculus", "statistics"}

// A Subdomain names one partition of a domain and holds its probing set size:
// the number of queries n drawn per query and the budget k for the Pass@K
// estimate. The probe set itself lives behind the Evaluator; Subdomain carries
// only the eval parameters the selector needs.
type Subdomain struct {
	Name     string // e.g. "algebra"
	NSamples int    // samples drawn per probe query (n)
	PassAtK  int    // Pass@K budget (k); requires 1 ≤ k ≤ NSamples
}

// An Evaluator scores a checkpoint on one subdomain's probe set and returns the
// mean Pass@K over that set. It is the injected seam by which a real model
// (sample n completions per query, verify, estimate Pass@K) plugs in; tests
// inject a deterministic fake, so no model is needed to exercise selection.
//
// The returned score is compared across checkpoints to pick the per-subdomain
// specialist; only its ordering matters to the selector.
type Evaluator interface {
	// Probe returns the mean Pass@K of checkpoint ckpt on subdomain sub. A
	// non-nil error aborts selection.
	Probe(ctx context.Context, ckpt string, sub Subdomain) (passK float64, err error)
}

// A Specialist is the checkpoint selected as the best for one subdomain, with
// the Pass@K score that won it. Step records which probing round (SFT step) the
// checkpoint came from, for provenance.
type Specialist struct {
	Subdomain  string
	Checkpoint string
	Step       int
	PassK      float64
}

// A Checkpoint is one SFT checkpoint offered to the probe: its training step
// and on-disk location. Selection scans the checkpoints in the given order; the
// step is recorded as provenance and breaks ties.
type Checkpoint struct {
	Step int
	Dir  string
}

// Select runs Domain-Aware Diversity Probing over a set of SFT checkpoints. For
// each subdomain Sᵢ it evaluates every checkpoint Mₜ on the subdomain's probe
// set with Pass@K and selects the diversity specialist
//
//	Mᵢ* = argmaxₜ Pᵢ(t),
//
// the checkpoint with the highest Pass@K on that subdomain — selecting for
// diversity, not lowest val loss or highest Pass@1. The returned specialists are
// in subdomain order, one per subdomain.
//
// Ties (equal Pass@K) are broken deterministically: the earliest checkpoint in
// the input order wins, and among equal scores a later checkpoint never
// displaces an earlier one. checkpoints and subs must be non-empty, eval must be
// non-nil, and each subdomain's PassAtK must lie in [1, NSamples].
func Select(ctx context.Context, eval Evaluator, checkpoints []Checkpoint, subs []Subdomain) ([]Specialist, error) {
	if eval == nil {
		return nil, fmt.Errorf("probe: nil evaluator")
	}
	if len(checkpoints) == 0 {
		return nil, fmt.Errorf("probe: no checkpoints to probe")
	}
	if len(subs) == 0 {
		return nil, fmt.Errorf("probe: no subdomains")
	}
	for _, s := range subs {
		if s.NSamples < 1 {
			return nil, fmt.Errorf("probe: subdomain %q NSamples must be >= 1, got %d", s.Name, s.NSamples)
		}
		if s.PassAtK < 1 || s.PassAtK > s.NSamples {
			return nil, fmt.Errorf("probe: subdomain %q PassAtK must be in [1, %d], got %d", s.Name, s.NSamples, s.PassAtK)
		}
	}
	return selectSpecialists(ctx, eval, checkpoints, subs)
}

// selectSpecialists is the unchecked core of Select. It assumes non-empty,
// validated inputs.
func selectSpecialists(ctx context.Context, eval Evaluator, checkpoints []Checkpoint, subs []Subdomain) ([]Specialist, error) {
	out := make([]Specialist, 0, len(subs))
	for _, sub := range subs {
		best := Specialist{Subdomain: sub.Name}
		found := false
		for _, ck := range checkpoints {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("probe: canceled probing %q: %w", sub.Name, err)
			}
			score, err := eval.Probe(ctx, ck.Dir, sub)
			if err != nil {
				return nil, fmt.Errorf("probe: subdomain %q checkpoint %q: %w", sub.Name, ck.Dir, err)
			}
			// Strict > keeps the earliest checkpoint on a tie: a later
			// checkpoint with an equal score does not displace it.
			if !found || score > best.PassK {
				best.Checkpoint = ck.Dir
				best.Step = ck.Step
				best.PassK = score
				found = true
			}
		}
		out = append(out, best)
	}
	return out, nil
}
