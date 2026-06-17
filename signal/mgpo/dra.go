package mgpo

import (
	"fmt"
	"math"
)

// DRA-GRPO diversity-aware reward adjustment (arXiv 2505.09655). Reasoning RL
// tends toward mode collapse: rollouts within a group grow similar, so reward
// concentrates on a few redundant modes and exploration dies — the same
// diversity loss VibeThinker's Spectrum phase exists to prevent. DRA reweights
// each rollout's reward by the inverse of its total similarity to its group
// siblings, before advantage normalization:
//
//	R_i ← R_i / Σ_j sim(o_i, o_j),
//
// over the O(G²) cosine-similarity matrix of the group's rollout embeddings. A
// rollout that is highly similar to the rest of the group (a crowded mode) is
// down-weighted; a distinctive rollout keeps more of its reward.
//
// The embedder is the only sanctioned external model in the upgrade, and it is
// an injected [Embedder] interface with an in-repo [FakeEmbedder], exactly like
// the teacher/rubric/sandbox seams — so this package builds and tests with zero
// external dependencies. A nil reweighter is DRA off (baseline). When every
// rollout in a group is identical, all pairwise similarities are 1, the divisor
// is the constant G for every rollout, and in the std-normalized advantage path
// the uniform rescale cancels — so an identity embedder reproduces the baseline.

// An Embedder maps rollout texts to fixed-length embedding vectors used to
// measure intra-group diversity. A real implementation wraps a sentence encoder
// (e.g. jina-embeddings-v2-small-en); the in-repo [FakeEmbedder] is
// deterministic and model-free. All embeddings returned for one call must share
// a length.
type Embedder interface {
	// Embed returns one vector per input text, in order.
	Embed(texts []string) ([][]float64, error)
}

// DiversityReweight rewrites a prompt group's rewards by DRA's inverse-similarity
// factor using the embeddings of the group's rollout texts. rewards and texts
// must have the same length (one per rollout). It returns the reweighted rewards
// in the same order.
//
// For each rollout i the divisor is Σ_j sim(o_i, o_j) where sim is cosine
// similarity clamped to [0,1] (negative cosines, which would inflate rather than
// damp a reward, are floored at 0; the self-term sim(o_i,o_i)=1 is included, so
// the divisor is ≥ 1 and the reweight is well defined). Identical rollouts give a
// uniform divisor G, so the std-normalized advantage is unchanged. A zero-norm
// embedding (no signal) contributes similarity 0 to others and 1 to itself.
func DiversityReweight(rewards []float64, texts []string, emb Embedder) ([]float64, error) {
	if emb == nil {
		return nil, fmt.Errorf("mgpo: DiversityReweight needs a non-nil Embedder")
	}
	if len(rewards) != len(texts) {
		return nil, fmt.Errorf("mgpo: DRA has %d rewards but %d rollout texts", len(rewards), len(texts))
	}
	n := len(rewards)
	if n == 0 {
		return nil, fmt.Errorf("mgpo: DRA on an empty group")
	}
	vecs, err := emb.Embed(texts)
	if err != nil {
		return nil, fmt.Errorf("mgpo: DRA embed: %w", err)
	}
	if len(vecs) != n {
		return nil, fmt.Errorf("mgpo: DRA embedder returned %d vectors for %d texts", len(vecs), n)
	}
	dim := len(vecs[0])
	for i, v := range vecs {
		if len(v) != dim {
			return nil, fmt.Errorf("mgpo: DRA embedding %d has length %d, want %d", i, len(v), dim)
		}
	}
	norms := make([]float64, n)
	for i, v := range vecs {
		norms[i] = math.Sqrt(dot(v, v))
	}
	out := make([]float64, n)
	for i := range rewards {
		var simSum float64
		for j := range vecs {
			simSum += cosine(i, j, vecs[i], vecs[j], norms[i], norms[j])
		}
		if simSum <= 0 {
			// Degenerate (all-zero embeddings): fall back to no reweight rather
			// than divide by zero. The self-term normally keeps simSum ≥ 1.
			out[i] = rewards[i]
			continue
		}
		out[i] = rewards[i] / simSum
	}
	return out, nil
}

// DiversityReweightGroups applies [DiversityReweight] to each prompt group of a
// batch, given the per-group rollout texts. It is the seam the advantage pipeline
// calls before [ScaledAdvantagesOpt]: the reweighted rewards feed group-relative
// advantage exactly as raw rewards do, so w_ME still multiplies the resulting
// advantage and the no-op rule holds. groupTexts[i] must align with rewards[i].
func DiversityReweightGroups(rewards [][]float64, groupTexts [][]string, emb Embedder) ([][]float64, error) {
	if len(rewards) != len(groupTexts) {
		return nil, fmt.Errorf("mgpo: DRA has %d reward groups but %d text groups", len(rewards), len(groupTexts))
	}
	out := make([][]float64, len(rewards))
	for i := range rewards {
		rw, err := DiversityReweight(rewards[i], groupTexts[i], emb)
		if err != nil {
			return nil, fmt.Errorf("mgpo: DRA group %d: %w", i, err)
		}
		out[i] = rw
	}
	return out, nil
}

func cosine(i, j int, a, b []float64, na, nb float64) float64 {
	if i == j {
		return 1 // self-similarity is exactly 1, even for a zero-norm embedding.
	}
	if na == 0 || nb == 0 {
		return 0 // a zero-norm vector is similar to nothing but itself.
	}
	c := dot(a, b) / (na * nb)
	if c < 0 {
		return 0 // floor negative cosines: they would inflate, not damp, a reward.
	}
	if c > 1 {
		return 1 // guard floating-point overshoot.
	}
	return c
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
