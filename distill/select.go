package distill

import "sort"

// A Trace is a verified rejection-sampled trace with its learning-potential
// score, its length in tokens, and the domain it came from.
type Trace struct {
	ID     string
	Domain string
	Length int
	Score  float64 // S_LP under the student
}

// SelectParams controls per-domain length-bucket selection (DESIGN §4.3).
type SelectParams struct {
	// MinLength drops extremely short traces before any selection.
	MinLength int

	// Buckets is the number of equal-width length buckets per domain.
	Buckets int

	// OutlierQuantile drops the top fraction of S_LP scores within each bucket
	// as high-score outliers (e.g. 0.1 drops the highest 10%). The remaining
	// middle-to-high range is kept.
	OutlierQuantile float64
}

// DefaultSelectParams returns reasonable defaults: drop traces under 64 tokens,
// 4 length buckets per domain, trim the top 10% of scores per bucket.
func DefaultSelectParams() SelectParams {
	return SelectParams{MinLength: 64, Buckets: 4, OutlierQuantile: 0.1}
}

// Select chooses the distillation set from verified traces. It groups traces by
// domain, drops traces shorter than MinLength, partitions each domain's traces
// into Buckets equal-width length buckets, trims the highest-scoring
// OutlierQuantile fraction within each bucket, and returns the union across all
// domains and buckets. Selection is per domain so a domain with uniformly long
// or short traces is bucketed on its own scale, not a global one.
//
// The returned traces are ordered by domain, then bucket, then descending
// score, so the most valuable kept traces lead each group.
func Select(traces []Trace, p SelectParams) []Trace {
	return selectCore(traces, p)
}

func selectCore(traces []Trace, p SelectParams) []Trace {
	buckets := max(p.Buckets, 1)
	byDomain := map[string][]Trace{}
	var domains []string
	for _, t := range traces {
		if t.Length < p.MinLength {
			continue
		}
		if _, ok := byDomain[t.Domain]; !ok {
			domains = append(domains, t.Domain)
		}
		byDomain[t.Domain] = append(byDomain[t.Domain], t)
	}
	sort.Strings(domains)

	var out []Trace
	for _, d := range domains {
		ts := byDomain[d]
		minLen, maxLen := lengthRange(ts)
		width := maxLen - minLen
		// Group into length buckets.
		grouped := make([][]Trace, buckets)
		for _, t := range ts {
			b := 0
			if width > 0 {
				b = (t.Length - minLen) * buckets / (width + 1)
				if b >= buckets {
					b = buckets - 1
				}
			}
			grouped[b] = append(grouped[b], t)
		}
		for _, g := range grouped {
			out = append(out, trimOutliers(g, p.OutlierQuantile)...)
		}
	}
	return out
}

// trimOutliers drops the highest-scoring quantile fraction of g and returns the
// rest sorted by descending score. A quantile <= 0 keeps everything; >= 1 drops
// everything but one (we never trim the whole bucket to empty when it has data).
func trimOutliers(g []Trace, quantile float64) []Trace {
	if len(g) == 0 {
		return nil
	}
	sorted := make([]Trace, len(g))
	copy(sorted, g)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Score != sorted[j].Score {
			return sorted[i].Score > sorted[j].Score // descending score
		}
		return sorted[i].ID < sorted[j].ID // deterministic tiebreak
	})
	if quantile <= 0 {
		return sorted
	}
	drop := int(float64(len(sorted)) * quantile)
	if drop >= len(sorted) {
		drop = len(sorted) - 1
	}
	// The highest scores lead the slice, so drop from the front.
	return sorted[drop:]
}

func lengthRange(ts []Trace) (lo, hi int) {
	lo, hi = ts[0].Length, ts[0].Length
	for _, t := range ts[1:] {
		lo = min(lo, t.Length)
		hi = max(hi, t.Length)
	}
	return lo, hi
}
