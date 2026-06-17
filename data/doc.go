// Package data provides VibeThinker's dataset loading, synthesis seams, and
// local quality control (DESIGN §4.6, §5.7).
//
// A [Sample] is the unit the pipeline trains and evaluates on: a prompt, its
// reference answer, the domain it belongs to (math, code, STEM, …), and the
// length of its reasoning trace in tokens. Datasets enter the pipeline through
// two seams:
//
//   - [Loader] reads samples from a public dataset or a bundled eval set. The
//     in-repo [SliceLoader] serves a fixed slice for tests and the toy pipeline.
//   - [Synthesizer] expands a seed set (concept composition, skeletons,
//     constraints, majority-vote pseudo-labeling). Real synthesis needs a
//     frontier teacher, so it stays a documented gate: this package defines the
//     interface only and supplies a deterministic [EchoSynthesizer] fake.
//
// Quality control runs locally. The paper drops degenerate traces — text that
// loops or repeats — before training. [RepetitionRate] measures the fraction of
// n-grams that are repeats,
//
//	rep_n = 1 − distinct_n / total_n,
//
// over the token n-grams of a sample's text; [QualityFilter] drops any sample
// whose rate at the configured n meets or exceeds a threshold. A sample with no
// repetition has rep_n = 0; a sample that repeats a single token forever has
// rep_n → 1. These bounds and the drop decision are pinned by the tests.
//
// [StratifyByLength] buckets samples into length quantiles for the two-stage
// curriculum (broad coverage → hard reasoning), so a stage can draw a
// length-balanced or length-targeted subset without a global sort each time.
//
// Decontamination against eval sets is a separate concern; see the subpackage
// [github.com/tmc/mlx-go-vibethinker/data/decontam].
package data
