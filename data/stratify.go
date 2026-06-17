package data

import (
	"fmt"
	"sort"
)

// A LengthStratum is one length bucket of the curriculum: its inclusive token
// length bounds and the samples that fall in it. Strata are returned in
// increasing length order, so Strata[0] is the shortest bucket.
type LengthStratum struct {
	MinLength int
	MaxLength int
	Samples   []Sample
}

// StratifyByLength partitions samples into buckets equal-count quantile strata
// by trace Length, for the two-stage curriculum (DESIGN §4.6: stratify by trace
// length for curriculum). Samples are sorted by length and split into buckets
// contiguous groups of as-equal-as-possible size, so each stratum holds roughly
// len(samples)/buckets samples regardless of the length distribution's shape.
//
// Sorting is stable on length with a deterministic tiebreak (prompt, then
// answer) so the result does not depend on input order. buckets must be
// positive. With fewer samples than buckets, trailing strata are empty but the
// bounds still partition the length axis.
func StratifyByLength(samples []Sample, buckets int) ([]LengthStratum, error) {
	if buckets <= 0 {
		return nil, fmt.Errorf("data: buckets must be positive, got %d", buckets)
	}
	return stratify(samples, buckets), nil
}

// stratify is the unchecked core of StratifyByLength.
func stratify(samples []Sample, buckets int) []LengthStratum {
	sorted := make([]Sample, len(samples))
	copy(sorted, samples)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Length != sorted[j].Length {
			return sorted[i].Length < sorted[j].Length
		}
		if sorted[i].Prompt != sorted[j].Prompt {
			return sorted[i].Prompt < sorted[j].Prompt
		}
		return sorted[i].Answer < sorted[j].Answer
	})

	out := make([]LengthStratum, buckets)
	n := len(sorted)
	for b := 0; b < buckets; b++ {
		lo := b * n / buckets
		hi := (b + 1) * n / buckets
		group := sorted[lo:hi]
		st := LengthStratum{Samples: append([]Sample(nil), group...)}
		if len(group) > 0 {
			st.MinLength = group[0].Length
			st.MaxLength = group[len(group)-1].Length
		}
		out[b] = st
	}
	return out
}
