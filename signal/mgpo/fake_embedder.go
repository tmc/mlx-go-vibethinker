package mgpo

// FakeEmbedder is a deterministic, model-free in-repo [Embedder] for tests and
// the toy pipeline. It maps each text to a fixed-length byte-histogram vector, so
// identical texts produce identical embeddings (cosine 1) and texts with
// different character content produce different embeddings — enough to exercise
// DRA's diversity reweight without any sentence encoder. It is not a substitute
// for a real embedder in a real run; the histogram captures lexical, not
// semantic, similarity.
type FakeEmbedder struct {
	// Dim is the embedding length (number of histogram buckets). Zero means the
	// default of 32.
	Dim int
}

// Embed returns one byte-histogram vector per text. Each text's bytes are
// bucketed modulo the embedding dimension; the vector counts bucket hits, so
// repeated identical texts map to identical vectors. An empty text maps to the
// zero vector (norm 0), which DRA treats as similar only to itself.
func (f FakeEmbedder) Embed(texts []string) ([][]float64, error) {
	dim := f.Dim
	if dim <= 0 {
		dim = 32
	}
	out := make([][]float64, len(texts))
	for i, t := range texts {
		v := make([]float64, dim)
		for k := 0; k < len(t); k++ {
			v[int(t[k])%dim]++
		}
		out[i] = v
	}
	return out, nil
}
