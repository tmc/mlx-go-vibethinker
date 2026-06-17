package data

import (
	"context"
	"fmt"
)

// A Sample is one training or evaluation item: a prompt, its reference Answer,
// the Domain it belongs to (e.g. "math", "code", "stem"), and the Length of its
// reasoning trace in tokens. Length is used by the curriculum to stratify by
// difficulty proxy; it is 0 when unknown.
type Sample struct {
	Prompt string
	Answer string
	Domain string
	Length int
}

// A Loader reads samples from a dataset: a public corpus or one of the bundled
// parquet eval sets. Load returns all samples in the dataset; a real loader
// streams or pages internally and materializes them. The context allows a real
// loader to honor cancellation while reading from disk or network.
type Loader interface {
	Load(ctx context.Context) ([]Sample, error)
}

// A Synthesizer expands a seed set into new samples by concept composition,
// skeleton instantiation, constraint injection, or majority-vote pseudo-
// labeling (DESIGN §4.6). Real synthesis needs a frontier teacher and is a
// documented gate; this interface is the seam a real run plugs into. Synthesize
// returns the synthesized samples (not the seeds).
type Synthesizer interface {
	Synthesize(ctx context.Context, seeds []Sample) ([]Sample, error)
}

// SliceLoader is an in-repo [Loader] over a fixed slice, for tests and the toy
// pipeline. Its zero value loads no samples.
type SliceLoader struct {
	Samples []Sample
}

// Load returns a copy of the loader's samples.
func (l SliceLoader) Load(ctx context.Context) ([]Sample, error) {
	out := make([]Sample, len(l.Samples))
	copy(out, l.Samples)
	return out, nil
}

// EchoSynthesizer is a deterministic in-repo [Synthesizer] for tests and the
// toy pipeline. For each seed it emits Copies near-duplicates with the prompt
// suffixed by a variant marker, so a run exercises the synthesis seam without a
// teacher. It is not a substitute for a real synthesizer in a real run.
type EchoSynthesizer struct {
	// Copies is the number of variants emitted per seed; defaults to 1 when
	// non-positive.
	Copies int
}

// Synthesize returns Copies variants of each seed.
func (s EchoSynthesizer) Synthesize(ctx context.Context, seeds []Sample) ([]Sample, error) {
	n := s.Copies
	if n <= 0 {
		n = 1
	}
	out := make([]Sample, 0, len(seeds)*n)
	for _, seed := range seeds {
		for i := 0; i < n; i++ {
			v := seed
			v.Prompt = fmt.Sprintf("%s [variant %d]", seed.Prompt, i)
			out = append(out, v)
		}
	}
	return out, nil
}
