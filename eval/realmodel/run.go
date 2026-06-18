//go:build modelir

package realmodel

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/tmc/mlx-go/mlx"
	"github.com/tmc/mlx-go/mlx/random"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
	"github.com/tmc/mlx-go-vibethinker/reward/mathverify"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// Config bounds the real-model smoke run. The defaults keep it a smoke (a few
// prompts, a few rollouts, short completions, a handful of real optimizer
// steps) — enough to prove each method's mechanism holds on real logits and
// that training stays stable, not to converge.
type Config struct {
	Prompts     int     // number of math prompt groups
	K           int     // rollouts per prompt (>=4 for real within-group spread)
	MaxTokens   int     // max generated tokens per rollout
	Temperature float64 // sampling temperature
	Steps       int     // real optimizer steps per method
	LR          float64 // learning rate
	Seed        uint64  // base RNG seed (rollouts only; the optimizer is not seedable)
}

// DefaultConfig returns a tractable smoke configuration for the 1.5B model on a
// single machine. MaxTokens and the prompt/rollout counts are bounded to stay
// under the Metal per-process array/command-buffer ceiling (~499000) seen on
// longer rollouts — this is a smoke, not a convergence run, and the bound is
// logged in the report header so the small config is not mistaken for a maximum.
func DefaultConfig() Config {
	return Config{
		Prompts:     6,
		K:           4,
		MaxTokens:   32,
		Temperature: 0.8,
		Steps:       8,
		LR:          1e-6,
		Seed:        1,
	}
}

// Source selects the rollout source for a run.
type Source int

const (
	// Organic generates rollouts from the live model — the honest, model-emitted
	// source. On a weak base the within-group reward spread collapses (acc≈0).
	Organic Source = iota
	// Seeded uses fixed, real-tokenized, real-Forward-rescored completions
	// constructed to guarantee mixed-correctness groups, so the reward-shape
	// mechanisms (Dr.GRPO/DCPO/Dynamic Sampling) are observable on real logits.
	// Only the completion TEXT is fixed; rescore + loss + optimizer step are real.
	Seeded
)

func (s Source) String() string {
	if s == Seeded {
		return "SEEDED"
	}
	return "ORGANIC"
}

// Evaluate runs the real-model smoke for every method in the registry, for the
// given rollout source, and returns their metrics in registry order. The model
// is loaded ONCE; before each method its base-policy weights are restored from a
// snapshot (the optimizer mutates the weights in place), so every method starts
// from the same policy without keeping a second multi-GB model resident — a
// fresh per-method reload OOMs the wired working set. The numbers are model- and
// machine-dependent and NOT reproducible across runs.
func Evaluate(ctx context.Context, dir string, cfg Config, src Source) ([]Metrics, error) {
	methods := methodcompare.Registry()
	out := make([]Metrics, 0, len(methods))
	for _, method := range methods {
		mt, err := evalOneMethod(ctx, dir, method, cfg, src)
		if err != nil {
			return nil, err
		}
		out = append(out, mt)
	}
	return out, nil
}

// MethodCount is the number of methods in the comparison registry.
func MethodCount() int { return len(methodcompare.Registry()) }

// MethodName returns the registry method name at idx (for naming an ERROR cell
// when the child trial fails before it can report its own metrics).
func MethodName(idx int) string {
	methods := methodcompare.Registry()
	if idx < 0 || idx >= len(methods) {
		return fmt.Sprintf("method#%d", idx)
	}
	return methods[idx].Name
}

// ParseSource maps "ORGANIC"/"organic"/"SEEDED"/"seeded" to a Source.
func ParseSource(s string) (Source, error) {
	switch s {
	case "ORGANIC", "organic":
		return Organic, nil
	case "SEEDED", "seeded":
		return Seeded, nil
	}
	return Organic, fmt.Errorf("realmodel: unknown source %q (want organic|seeded)", s)
}

// EvaluateOneMethod runs a SINGLE method (by registry index) under one source
// and returns its metrics. It exists for the subprocess-per-method harness: the
// real value-and-grad over the 1.5B retains ~13GB of compiled VG/activation
// graph per method that the substrate's separate-VG path does not release back
// to the Metal allocator, so running all nine methods in one process trips the
// device resource ceiling (~499000 live buffers). Running each method in its own
// OS process — which this entry enables — lets that memory die with the child.
func EvaluateOneMethod(ctx context.Context, dir string, idx int, cfg Config, src Source) (Metrics, error) {
	methods := methodcompare.Registry()
	if idx < 0 || idx >= len(methods) {
		return Metrics{}, fmt.Errorf("realmodel: method index %d out of range [0,%d)", idx, len(methods))
	}
	return evalOneMethod(ctx, dir, methods[idx], cfg, src)
}

// evalOneMethod loads a fresh model, runs one method to completion, and releases
// the model before returning.
func evalOneMethod(ctx context.Context, dir string, method Method, cfg Config, src Source) (mt Metrics, err error) {
	m, err := Load(ctx, dir)
	if err != nil {
		return Metrics{}, fmt.Errorf("realmodel: load for %q: %w", method.Name, err)
	}
	defer func() {
		m.Close()
		reclaim()
	}()
	groups, err := buildGroups(ctx, m, cfg, src)
	if err != nil {
		return Metrics{}, fmt.Errorf("realmodel: build %s groups for %q: %w", src, method.Name, err)
	}
	defer freeGroups(groups)
	mt, err = runMethod(ctx, m, method, cfg, groups)
	if err != nil {
		return Metrics{}, fmt.Errorf("realmodel: run %q (%s): %w", method.Name, src, err)
	}
	mt.Source = src.String()
	return mt, nil
}

// freeGroups releases the frozen old/ref snapshot arrays held by a method's
// rollout groups.
func freeGroups(groups []group) {
	for _, g := range groups {
		if g.old.logProbs != nil {
			g.old.logProbs.Free()
		}
		if g.old.mask != nil {
			g.old.mask.Free()
		}
		// ref shares old's mask; free only ref's distinct logProbs.
		if g.ref.logProbs != nil && g.ref.logProbs != g.old.logProbs {
			g.ref.logProbs.Free()
		}
	}
}

// buildGroups dispatches to the organic or seeded group builder.
func buildGroups(ctx context.Context, m *Model, cfg Config, src Source) ([]group, error) {
	if src == Seeded {
		return buildSeededGroups(ctx, m, cfg)
	}
	return buildOrganicGroups(ctx, m, cfg)
}

// EvaluateBoth runs the smoke for every method under BOTH the organic and seeded
// rollout sources, returning the organic metrics then the seeded metrics. The
// organic block is the honest model-generated evidence (mechanism + stability +
// the load-bearing ratio stats); the seeded block exhibits the reward-shape
// mechanisms (Dr.GRPO/DCPO/Dynamic Sampling) that a ~0%-accuracy base cannot
// surface organically. Both are real-logit, real-optimizer runs.
func EvaluateBoth(ctx context.Context, dir string, cfg Config) (organic, seeded []Metrics, err error) {
	organic, err = Evaluate(ctx, dir, cfg, Organic)
	if err != nil {
		return nil, nil, err
	}
	seeded, err = Evaluate(ctx, dir, cfg, Seeded)
	if err != nil {
		return nil, nil, err
	}
	return organic, seeded, nil
}

// group is one prompt's rollouts with their scored rewards and the frozen
// old/ref policy snapshots captured at rollout time (before any optimizer step).
type group struct {
	prompt   mathPrompt
	rollouts []rollout
	rewards  []float64 // one per rollout
	promptID string
	texts    []string // rollout completion texts (for DRA)
	old, ref scored   // frozen behavior + reference snapshots
}

// scoreAndFreeze scores a group's rollouts against the gold answer and captures
// the frozen old/ref policy snapshots from the current model — the shared tail
// of both the organic and seeded group builders. The rollouts must be non-empty
// and length >= 2.
func scoreAndFreeze(ctx context.Context, m *Model, p mathPrompt, gi int, rollouts []rollout) (group, error) {
	rewards := make([]float64, len(rollouts))
	texts := make([]string, len(rollouts))
	for i, r := range rollouts {
		rewards[i] = mathverify.Reward(r.text, p.Gold)
		texts[i] = r.text
	}
	// old (behavior) and ref (reference) are frozen snapshots from the current
	// (base) policy, captured before any optimizer step.
	oldScored, err := rescoreGroup(ctx, m.LM, rollouts)
	if err != nil {
		return group{}, fmt.Errorf("realmodel: rescore group %d: %w", gi, err)
	}
	oldFrozen, err := captureFrozen(oldScored)
	if err != nil {
		return group{}, err
	}
	refFrozen, err := captureFrozen(oldScored)
	if err != nil {
		return group{}, err
	}
	return group{
		prompt:   p,
		rollouts: rollouts,
		rewards:  rewards,
		promptID: fmt.Sprintf("p%d", gi),
		texts:    texts,
		old:      oldFrozen,
		ref:      refFrozen,
	}, nil
}

// buildOrganicGroups generates rollouts from the live model for each prompt and
// scores them — the honest, model-generated rollout source. Groups that come
// back empty or degenerate (< 2 usable rollouts) are skipped.
func buildOrganicGroups(ctx context.Context, m *Model, cfg Config) ([]group, error) {
	baseKey := random.Key(cfg.Seed)
	prompts := mathPrompts()
	if cfg.Prompts < len(prompts) {
		prompts = prompts[:cfg.Prompts]
	}
	groups := make([]group, 0, len(prompts))
	for gi, p := range prompts {
		nextKey, gKey := random.Split(baseKey, nil)
		baseKey = nextKey
		rollouts, err := m.rolloutGroup(ctx, p, cfg.K, cfg.MaxTokens, cfg.Temperature, gKey)
		if err != nil {
			return nil, fmt.Errorf("realmodel: rollout group %d: %w", gi, err)
		}
		rollouts = nonEmpty(rollouts)
		if len(rollouts) < 2 {
			continue
		}
		g, err := scoreAndFreeze(ctx, m, p, gi, rollouts)
		if err != nil {
			return nil, err
		}
		groups = append(groups, g)
		reclaim()
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("realmodel: no usable organic rollout groups")
	}
	return groups, nil
}

// reclaim drains in-flight command buffers, runs the GC twice to fire the
// mlx.Array finalizers (which Free the underlying Metal buffers) and reclaim
// what they freed, then drops the buffer cache. Called between rollout groups
// and methods so the live Metal-resource count stays under the device ceiling
// (~499000) over a long multi-method run.
func reclaim() {
	mlx.Synchronize()
	runtime.GC()
	runtime.GC()
	mlx.ClearCache()
}

// runMethod runs the full real GRPO smoke for one method over the given source
// of rollout groups (organic or seeded). Each step recomputes `current` from the
// live (updated) weights against the group's frozen old/ref, so after >=1 step
// the importance ratio genuinely deviates from 1 — the clip/FRPO terms do real
// work, not the ratio==1 artifact. It records the real-logit mechanism metrics.
//
// The model is reloaded fresh per method by the caller so each method starts
// from the same base policy (the optimizer mutates the weights in place).
func runMethod(ctx context.Context, m *Model, method Method, cfg Config, groups []group) (Metrics, error) {
	mt := Metrics{Method: method.Name}
	start := time.Now()

	if len(groups) == 0 {
		return mt, fmt.Errorf("realmodel: no rollout groups for method %q", method.Name)
	}

	// --- Pre-step mechanism metrics over the base-policy rollouts. ---
	recordAdvantageMetrics(&mt, groups, method)
	recordRewardMetrics(&mt, groups, method)
	recordFutureKL(&mt, ctx, m, groups, method)

	cfg2 := baseConfig()
	cfg2.DrGRPO = method.DrGRPOLoss

	// --- Real optimizer steps. ---
	// One trainer drives all steps; each step's loss closure rescore-recomputes
	// `current` from the live weights for one cycling group, against that group's
	// frozen old/ref. The DCPO running-stats store persists across steps.
	var stats *mgpo.PromptStats
	if method.DCPOSmoothing {
		stats = mgpo.NewPromptStats()
	}

	// activeGroup holds the per-step closure state; the loss closure reads it.
	var active group
	var activeRewards [][]float64
	var activeIDs []string
	var lossClosure func() (*mlx.Array, error)
	tr, err := newTrainer(m, cfg.LR, func() (*mlx.Array, error) { return lossClosure() })
	if err != nil {
		return mt, err
	}
	defer tr.free()

	lossClosure = func() (*mlx.Array, error) {
		// `current` from the LIVE weights (the trainer wrote params into the model
		// before calling this) — the only gradient-carrying term. old/ref are the
		// frozen snapshots captured at rollout time.
		cur, err := rescoreGroup(ctx, m.LM, active.rollouts)
		if err != nil {
			return nil, err
		}
		return grpoLossDCPO(cur.logProbs, active.old.logProbs, active.ref.logProbs, cur.mask, activeRewards, method, cfg2, stats, activeIDs)
	}

	losses := make([]float64, 0, cfg.Steps)
	var ratioMeanSum, ratioVarSum, ratioMaxAbsDev, clipBindHighMax float64
	var ratioSamples int

	curGroup := 0
	for stepIdx := 0; stepIdx < cfg.Steps; stepIdx++ {
		g := groups[curGroup%len(groups)]
		curGroup++

		// This step's method-adjusted rewards (DRA reweight, Dynamic Sampling).
		rewards2D := [][]float64{append([]float64(nil), g.rewards...)}
		ids := []string{g.promptID}
		texts2D := [][]string{g.texts}
		if method.DRA != nil {
			rw, err := mgpo.DiversityReweightGroups(rewards2D, texts2D, method.DRA)
			if err != nil {
				return mt, err
			}
			rewards2D = rw
		}
		if method.DynamicSampling {
			rewards2D, ids = mgpo.DynamicSample(rewards2D, ids)
		}
		if len(rewards2D) == 0 {
			// Dynamic Sampling dropped this zero-gradient group (acc∈{0,1}): there
			// is nothing to learn from it, so skip the step entirely — the same
			// no-op outcome as the std=0 zero-advantage guard. Count it as a real
			// (skipped) step so GroupsDropped is observable.
			mt.GroupsDropped++
			continue
		}

		active, activeRewards, activeIDs = g, rewards2D, ids
		loss, err := tr.step()
		if err != nil {
			return mt, fmt.Errorf("realmodel: step %d (%s): %w", stepIdx, method.Name, err)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return mt, fmt.Errorf("realmodel: non-finite loss %v at step %d (%s)", loss, stepIdx, method.Name)
		}
		losses = append(losses, loss)

		// After the step, measure the importance ratio current/old on this group:
		// direct evidence the rescore is live (ratio not a delta at 1) and the
		// real clip-bind rate.
		rmean, rvar, rmad, chigh, ok := ratioStats(ctx, m, g, baseConfig(), method)
		if ok {
			ratioMeanSum += rmean
			ratioVarSum += rvar
			ratioMaxAbsDev = math.Max(ratioMaxAbsDev, rmad)
			clipBindHighMax = math.Max(clipBindHighMax, chigh)
			ratioSamples++
		}
		reclaim()
	}

	mt.Steps = len(losses)
	if len(losses) == 0 {
		// All groups were dropped — for +DynSampling on an all-cliff set (every
		// group acc∈{0,1}, which a ~0%-accuracy base produces organically) this is
		// the CORRECT mechanism: Dynamic Sampling removes every zero-gradient group,
		// so there is nothing to step on. Record it honestly: 0 steps, no loss
		// (JSON null), no divergence (there was nothing to diverge), the drop count
		// telling the story. Not an error.
		mt.LossFinite = true
		mt.WallMillis = float64(time.Since(start).Milliseconds())
		return mt, nil
	}
	mt.FinalLoss = ptr(losses[len(losses)-1])
	mt.LossFinite = true
	mt.MaxAbsLoss = maxAbs(losses)
	if ratioSamples > 0 {
		mt.RatioMean = ratioMeanSum / float64(ratioSamples)
		mt.RatioVar = ratioVarSum / float64(ratioSamples)
		mt.RatioMaxAbsDev = ratioMaxAbsDev
		mt.ClipBindHigh = clipBindHighMax
	}
	mt.WallMillis = float64(time.Since(start).Milliseconds())
	return mt, nil
}

// ptr returns a pointer to v (FinalLoss is *float64 so "no loss" is JSON null).
func ptr(v float64) *float64 { return &v }

// nonEmpty drops rollouts with empty completions.
func nonEmpty(rs []rollout) []rollout {
	out := rs[:0]
	for _, r := range rs {
		if len(r.completion) > 0 {
			out = append(out, r)
		}
	}
	return out
}

func maxAbs(xs []float64) float64 {
	var m float64
	for _, x := range xs {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}
