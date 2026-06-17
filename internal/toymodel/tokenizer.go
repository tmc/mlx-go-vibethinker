package toymodel

// Tokenizer is a byte-level tokenizer: each byte maps to one token id in
// [0,256). It is deterministic and reversible, adequate for exercising the
// data, packing, and log-probability code paths in tests.
type Tokenizer struct{}

// Encode returns the byte values of s as token ids.
func (Tokenizer) Encode(s string) []int {
	b := []byte(s)
	ids := make([]int, len(b))
	for i, c := range b {
		ids[i] = int(c)
	}
	return ids
}

// Decode reconstructs the string from token ids, ignoring ids outside [0,256).
func (Tokenizer) Decode(ids []int) string {
	b := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < 256 {
			b = append(b, byte(id))
		}
	}
	return string(b)
}

// VocabSize reports the number of distinct token ids.
func (Tokenizer) VocabSize() int { return 256 }
