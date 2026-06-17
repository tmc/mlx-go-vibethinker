// Package methodcompare is a method-comparison harness for the post-GRPO
// upgrades in signal/mgpo (DESIGN_RL_UPGRADE.md). It enumerates named
// configurations — the DESIGN.md baseline MGPO and each Tier-1/2/3 refinement,
// plus the stacked all-on — runs every one on a fixed, deterministic synthetic
// scenario, and reports the mechanism metrics that the theory predicts each knob
// should move.
//
// IMPORTANT: this is a TOY substrate. The harness measures MECHANISM, not
// benchmark accuracy: whether a knob moves the metric its paper predicts
// (Dr.GRPO removes the std divisor from |A|; Clip-Higher raises the upper
// clip-bind rate; Long2Short cuts tokens per sample at equal reward; DCPO-SAS
// smooths advantage variance across steps). The numbers are synthetic-scenario
// deltas, NOT paper accuracy deltas — do not read a toy delta as a benchmark
// result. The table and JSON headers say so.
//
// The metrics here are pure, deterministic functions of a fixed scenario (a
// seed picks the scenario; the same seed yields identical numbers), so they need
// no model and the reproducibility test runs tag-free. The model-pipeline
// metrics (per-stage loss, wall-time) are wired onto the real toy 1.5B/3B
// pipeline behind the modelir build tag in run_modelir.go.
package methodcompare

import (
	"fmt"
	"math"
	"sort"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"

	"github.com/tmc/mlx-go-vibethinker/signal/long2short"
	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// A Method is one named configuration in the comparison: the DESIGN.md baseline
// MGPO with a specific set of post-GRPO refinements enabled. The zero Method
// (only a name) is the baseline; each field opts in one refinement, mirroring the
// signal/mgpo option surface so the harness configuration is exactly the training
// configuration.
type Method struct {
	Name string

	// Opts selects the Tier-1 advantage/clip refinements (Dr.GRPO advantage,
	// Clip-Higher eps). The zero Options is the baseline.
	Opts mgpo.Options

	// Lambda is the MGPO max-entropy coefficient (w_ME). The shared default is
	// used when zero is not meaningful; the registry sets it explicitly.
	Lambda float64

	// DrGRPOLoss sets the substrate length-norm half of Dr.GRPO (GRPOConfig.DrGRPO)
	// for the loss metric. The advantage half is Opts.DrGRPOAdvantage.
	DrGRPOLoss bool

	// DCPOSmoothing enables DCPO Smooth Advantage Standardization across the
	// scenario's steps (a per-prompt running-stats store).
	DCPOSmoothing bool

	// DynamicSampling enables the DAPO group filter (drop acc in {0,1}).
	DynamicSampling bool

	// FRPO enables the future-KL sibling loss with this BetaFuture (0 = off).
	FRPO mgpo.FRPOConfig

	// HDPO enables the cliff-JSD term with this LambdaJSD (0 = off).
	HDPO mgpo.HDPOConfig

	// DRA enables the diversity-aware reward reweight with the given embedder
	// (nil = off).
	DRA mgpo.Embedder
}

// Metrics holds one method's measured mechanism metrics over the scenario. The
// model-pipeline metrics are filled in by the modelir layer; the tag-free core
// fills the rest. All fields are deterministic for a fixed scenario.
type Metrics struct {
	Method string `json:"method"`

	// Advantage statistics over the final step's modulated advantages.
	AdvMean    float64 `json:"adv_mean"`
	AdvStd     float64 `json:"adv_std"`
	AdvAbsMean float64 `json:"adv_abs_mean"`

	// w_ME distribution over the scenario's prompt groups.
	WMEMean float64 `json:"wme_mean"`
	WMEMin  float64 `json:"wme_min"`
	WMEMax  float64 `json:"wme_max"`

	// ClipBindRate is the fraction of per-token importance ratios that fall
	// outside the (asymmetric) clip range — the tokens whose surrogate is
	// clipped. Clip-Higher raises the upper clip ceiling (1+ε_high), so FEWER
	// high-side ratios bind: it gives low-probability exploration tokens more
	// room to grow before clipping, which is the entropy-preserving effect. The
	// expected delta is therefore a LOWER ClipBindHigh, not a higher one.
	ClipBindRate float64 `json:"clip_bind_rate"`
	ClipBindHigh float64 `json:"clip_bind_high"` // fraction bound on the upper side

	// TokensPerSampleRaw is the uniform-mean trace length over the correct set
	// BEFORE Long2Short; TokensPerSample is the reward-weighted mean length AFTER
	// Long2Short reshaping. Long2Short shifts reward toward shorter correct traces
	// at equal total reward, so TokensPerSample < TokensPerSampleRaw — fewer
	// tokens per sample at the same group reward sum. (These are scenario
	// constants; Long2Short is always applied, so the raw→reshaped delta is the
	// mechanism, identical across method rows.)
	TokensPerSampleRaw float64 `json:"tokens_per_sample_raw"`
	TokensPerSample    float64 `json:"tokens_per_sample"`

	// AdvVarAcrossSteps is the mean (over rollout slots) of the variance of the
	// modulated advantage across the scenario's steps. DCPO-SAS smooths the
	// per-prompt advantage across steps, lowering this.
	AdvVarAcrossSteps float64 `json:"adv_var_across_steps"`

	// GroupsKept is the number of prompt groups surviving the data-layer filter
	// (Dynamic Sampling drops zero-gradient groups). Baseline keeps all.
	GroupsKept int `json:"groups_kept"`
	GroupsIn   int `json:"groups_in"`

	// CliffGroups is the number of zero-reward groups in the scenario (HDPO's
	// active set); CliffJSDTerm is the HDPO loss contribution (0 when off).
	CliffGroups  int     `json:"cliff_groups"`
	CliffJSDTerm float64 `json:"cliff_jsd_term"`

	// FinalLoss and the per-stage losses come from the toy pipeline (modelir
	// layer). FinalLoss is nil (JSON null) when the harness runs tag-free (no
	// model) — "no loss measured", not a numeric zero. These fields are
	// model/machine-dependent and NOT reproducible (see metric_layers).
	FinalLoss  *float64           `json:"final_loss"`
	StageLoss  map[string]float64 `json:"stage_loss,omitempty"`
	WallMillis float64            `json:"wall_millis"`
}

// Registry returns the named configurations to compare, in display order:
// baseline MGPO, each single refinement, then the stacked all-on. The lambda is
// shared so the w_ME column is comparable across rows.
func Registry() []Method {
	const lambda = 1.0
	emb := mgpo.FakeEmbedder{}
	return []Method{
		{Name: "baseline", Lambda: lambda},
		{Name: "+DrGRPO", Lambda: lambda, Opts: mgpo.Options{DrGRPOAdvantage: true}, DrGRPOLoss: true},
		{Name: "+ClipHigher", Lambda: lambda, Opts: mgpo.Options{ClipEpsLow: 0.2, ClipEpsHigh: 0.28}},
		{Name: "+DCPO-SAS", Lambda: lambda, DCPOSmoothing: true},
		{Name: "+DynSampling", Lambda: lambda, DynamicSampling: true},
		{Name: "+FRPO", Lambda: lambda, FRPO: mgpo.FRPOConfig{BetaFuture: 0.1}},
		{Name: "+HDPO", Lambda: lambda, HDPO: mgpo.HDPOConfig{LambdaJSD: 0.5}},
		{Name: "+DRA", Lambda: lambda, DRA: emb},
		{
			Name:            "all-on",
			Lambda:          lambda,
			Opts:            mgpo.Options{DrGRPOAdvantage: true, ClipEpsLow: 0.2, ClipEpsHigh: 0.28},
			DrGRPOLoss:      true,
			DCPOSmoothing:   true,
			DynamicSampling: true,
			FRPO:            mgpo.FRPOConfig{BetaFuture: 0.1},
			HDPO:            mgpo.HDPOConfig{LambdaJSD: 0.5},
			DRA:             emb,
		},
	}
}

// Evaluate runs every method in the registry on the scenario for the seed and
// returns their mechanism metrics in registry order. It is deterministic: the
// same seed yields identical metrics (the reproducibility contract). The
// model-pipeline fields (FinalLoss, StageLoss, WallMillis) are left at their
// tag-free defaults here; the modelir layer fills them.
func Evaluate(seed uint64) ([]Metrics, error) {
	sc := newScenario(seed)
	methods := Registry()
	out := make([]Metrics, len(methods))
	for i, m := range methods {
		mt, err := evalMechanism(m, sc)
		if err != nil {
			return nil, fmt.Errorf("methodcompare: method %q: %w", m.Name, err)
		}
		out[i] = mt
	}
	return out, nil
}

// evalMechanism computes the tag-free mechanism metrics for one method over the
// scenario. Every metric is a pure function of the fixed scenario and the
// method's configuration.
func evalMechanism(m Method, sc *scenario) (Metrics, error) {
	mt := Metrics{Method: m.Name} // FinalLoss nil: no model in the tag-free core

	// --- Per-step modulated advantages, with the method's full advantage path. ---
	stats := (*mgpo.PromptStats)(nil)
	if m.DCPOSmoothing {
		stats = mgpo.NewPromptStats()
	}
	stepAdv := make([][][]float64, len(sc.steps)) // step -> group -> slot
	var finalKeptRewards [][]float64              // rewards fed to advantage on the last step
	for s, step := range sc.steps {
		rewards := step.rewards
		ids := step.promptIDs

		// DRA diversity reweight (before advantage normalization).
		if m.DRA != nil {
			rw, err := mgpo.DiversityReweightGroups(rewards, step.texts, m.DRA)
			if err != nil {
				return mt, err
			}
			rewards = rw
		}

		// DAPO Dynamic Sampling (data-layer group filter).
		if m.DynamicSampling {
			rewards, ids = mgpo.DynamicSample(rewards, ids)
		}

		var adv [][]float64
		var err error
		if stats != nil {
			adv, err = mgpo.ScaledAdvantagesStep(rewards, m.Lambda, m.Opts, stats, ids)
		} else {
			adv, err = mgpo.ScaledAdvantagesOpt(rewards, m.Lambda, m.Opts)
		}
		if err != nil {
			return mt, err
		}
		stepAdv[s] = adv
		if s == len(sc.steps)-1 {
			mt.GroupsIn = len(step.rewards)
			mt.GroupsKept = len(rewards)
			finalKeptRewards = rewards
		}
	}

	// --- Advantage statistics over the final step. ---
	final := flatten(stepAdv[len(stepAdv)-1])
	mt.AdvMean, mt.AdvStd, mt.AdvAbsMean = meanStdAbs(final)

	// --- Advantage variance across steps (DCPO smoothing target). ---
	mt.AdvVarAcrossSteps = advVarAcrossSteps(stepAdv)

	// --- w_ME distribution over the final step's KEPT groups (after Dynamic
	// Sampling, which drops the acc∈{0,1} groups whose w_ME is at the extreme,
	// so the mean rises when the filter is on). ---
	mt.WMEMean, mt.WMEMin, mt.WMEMax = wMEStats(finalKeptRewards, m.Lambda)

	// --- Clip-bind rate over the scenario's per-token ratios. ---
	low, high := clipRange(m)
	mt.ClipBindRate, mt.ClipBindHigh = clipBindRates(sc.ratios, low, high)

	// --- Tokens per sample: raw (pre-reshape) vs after Long2Short. ---
	mt.TokensPerSampleRaw = rawTokensPerSample(sc.traces)
	tps, err := tokensPerSample(sc.traces)
	if err != nil {
		return mt, err
	}
	mt.TokensPerSample = tps

	// --- Cliff set / HDPO term over the final step. ---
	lastRewards := sc.steps[len(sc.steps)-1].rewards
	cliff := mgpo.CliffSet(lastRewards)
	mt.CliffGroups = len(cliff)
	if m.HDPO.LambdaJSD != 0 && len(cliff) > 0 {
		jsd := sc.cliffJSD(lastRewards)
		term, err := mgpo.HDPOLossTerm(0, lastRewards, jsd, m.HDPO)
		if err != nil {
			return mt, err
		}
		mt.CliffJSDTerm = term // base 0 ⇒ this is exactly λ·mean(JSD over C)
	}

	return mt, nil
}

// baselineClipEps is the harness's symmetric baseline clip epsilon. The DESIGN.md
// baseline clips symmetrically; rl.DefaultGRPOConfig already ships the asymmetric
// Clip-Higher values (0.2/0.28), so the harness must NOT use it as the "baseline"
// — that would make the +ClipHigher row indistinguishable from baseline. We build
// the per-method config from a symmetric base and let mgpo.Options.ClipRange (the
// +ClipHigher row) override it, so the clip-bind delta is real.
const baselineClipEps = 0.2

// baseConfig returns the symmetric baseline GRPOConfig the harness layers method
// options onto: every field is the substrate default except the clip, which is
// forced symmetric (ClipEps and both sides = baselineClipEps) so an unset Options
// yields a genuinely symmetric clip.
func baseConfig() rl.GRPOConfig {
	cfg := rl.DefaultGRPOConfig()
	cfg.ClipEps = baselineClipEps
	cfg.ClipEpsLow = baselineClipEps
	cfg.ClipEpsHigh = baselineClipEps
	return cfg
}

// clipRange resolves the method's effective clip range against the symmetric
// baseline config, applying the substrate's zero→ClipEps fallback exactly as
// rl.GRPOLoss does (via mgpo.Options.ClipRange).
func clipRange(m Method) (low, high float64) {
	return m.Opts.ClipRange(baseConfig())
}

// clipBindRates returns the fraction of importance ratios outside [1-low, 1+high]
// (bound either side) and the fraction bound specifically on the upper side.
func clipBindRates(ratios []float64, low, high float64) (rate, highRate float64) {
	if len(ratios) == 0 {
		return 0, 0
	}
	loBound := 1 - low
	hiBound := 1 + high
	var bound, boundHigh int
	for _, r := range ratios {
		if r > hiBound {
			bound++
			boundHigh++
		} else if r < loBound {
			bound++
		}
	}
	n := float64(len(ratios))
	return float64(bound) / n, float64(boundHigh) / n
}

// tokensPerSample is the reward-weighted mean length over the correct set after
// Long2Short reshaping: Σ_C w_i·L_i / Σ_C w_i, where w_i is the reshaped reward.
// Long2Short raises the reshaped reward of shorter correct traces, lowering this
// without changing the group reward sum (the reshape is zero-sum over C).
func tokensPerSample(traces []long2short.Trace) (float64, error) {
	reshaped, err := long2short.Reshape(traces, long2short.DefaultLambda)
	if err != nil {
		return 0, err
	}
	var num, den float64
	for i, tr := range traces {
		if !tr.Correct {
			continue
		}
		num += reshaped[i] * float64(tr.Length)
		den += reshaped[i]
	}
	if den == 0 {
		return 0, nil
	}
	return num / den, nil
}

// rawTokensPerSample is the uniform (un-reshaped, equal-weight) mean length over
// the correct set — the tokens-per-sample before Long2Short. Comparing it to
// tokensPerSample shows the brevity shift Long2Short applies at equal reward.
func rawTokensPerSample(traces []long2short.Trace) float64 {
	var sum float64
	var n int
	for _, tr := range traces {
		if !tr.Correct {
			continue
		}
		sum += float64(tr.Length)
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func wMEStats(rewards [][]float64, lambda float64) (mean, min, max float64) {
	if len(rewards) == 0 {
		return 0, 0, 0
	}
	min = math.Inf(1)
	max = math.Inf(-1)
	var sum float64
	for _, g := range rewards {
		w, _ := mgpo.Weight(lambda, mgpo.Accuracy(g))
		sum += w
		if w < min {
			min = w
		}
		if w > max {
			max = w
		}
	}
	return sum / float64(len(rewards)), min, max
}

func advVarAcrossSteps(stepAdv [][][]float64) float64 {
	if len(stepAdv) == 0 {
		return 0
	}
	// Align by (group, slot) and accumulate in a DETERMINISTIC order: iterate
	// groups then slots in index order (never a map, whose iteration order Go
	// randomizes and which would make the variance — and thus the whole table —
	// non-reproducible across runs). A slot must be present in every step to be
	// counted; the first step's shape is the reference.
	ref := stepAdv[0]
	var sum float64
	var n int
	for g := range ref {
		for j := range ref[g] {
			series := make([]float64, 0, len(stepAdv))
			ok := true
			for _, step := range stepAdv {
				if g >= len(step) || j >= len(step[g]) {
					ok = false
					break
				}
				series = append(series, step[g][j])
			}
			if !ok {
				continue // slot not present in every step (e.g. filtered)
			}
			_, std, _ := meanStdAbs(series)
			sum += std * std
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func flatten(groups [][]float64) []float64 {
	var out []float64
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

func meanStdAbs(xs []float64) (mean, std, absMean float64) {
	if len(xs) == 0 {
		return 0, 0, 0
	}
	var sum, absSum float64
	for _, x := range xs {
		sum += x
		absSum += math.Abs(x)
	}
	mean = sum / float64(len(xs))
	absMean = absSum / float64(len(xs))
	var sq float64
	for _, x := range xs {
		d := x - mean
		sq += d * d
	}
	std = math.Sqrt(sq / float64(len(xs)))
	return mean, std, absMean
}

// sortedMethodNames returns the registry method names in display order, for the
// emitters and tests.
func sortedMethodNames() []string {
	methods := Registry()
	names := make([]string, len(methods))
	for i, m := range methods {
		names[i] = m.Name
	}
	return names
}

// stableKeys returns a stage-loss map's keys in sorted order for deterministic
// table/JSON emission.
func stableKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
