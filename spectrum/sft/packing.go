package sft

import (
	"fmt"

	"github.com/tmc/mlx-go/mlx"
)

// A Packed block is the result of concatenating several token sequences into a
// single fixed-length row. It records where each original sequence (segment)
// begins so that downstream code can build a block-diagonal attention mask.
type Packed struct {
	// Tokens is the concatenated token ids, length BlockSize. Trailing unused
	// positions (when the segments do not fill the block) hold PadID and are
	// excluded by the loss mask.
	Tokens []int

	// SegmentLengths gives the length of each packed segment, in order. Their
	// sum is the number of real (non-pad) tokens in the block.
	SegmentLengths []int

	// SegmentID[t] is the index of the segment occupying position t, or -1 for
	// padding positions. It has length BlockSize.
	SegmentID []int
}

// PackResult holds the packed blocks produced from a set of sequences.
type PackResult struct {
	Blocks    []Packed
	BlockSize int
	PadID     int
}

// Pack concatenates sequences into fixed-length blocks of blockSize tokens
// using a first-fit greedy strategy: each sequence is appended to the current
// block if it fits, otherwise the current block is flushed and a new one
// started. A sequence longer than blockSize is an error (the caller must
// truncate or split upstream). Empty sequences are skipped.
//
// Pack performs no MLX work; it is a pure data transform. Use [BlockMask] to
// build the block-diagonal attention mask for a packed block.
func Pack(sequences [][]int, blockSize, padID int) (*PackResult, error) {
	if blockSize <= 0 {
		return nil, fmt.Errorf("sft: blockSize must be positive, got %d", blockSize)
	}
	res := &PackResult{BlockSize: blockSize, PadID: padID}
	var cur []int
	var curLens []int

	flush := func() {
		if len(cur) == 0 {
			return
		}
		res.Blocks = append(res.Blocks, finishBlock(cur, curLens, blockSize, padID))
		cur = nil
		curLens = nil
	}

	for i, seq := range sequences {
		if len(seq) == 0 {
			continue
		}
		if len(seq) > blockSize {
			return nil, fmt.Errorf("sft: sequence %d has length %d > blockSize %d", i, len(seq), blockSize)
		}
		if len(cur)+len(seq) > blockSize {
			flush()
		}
		cur = append(cur, seq...)
		curLens = append(curLens, len(seq))
	}
	flush()
	return res, nil
}

// finishBlock pads cur to blockSize and computes its segment ids.
func finishBlock(tokens []int, lens []int, blockSize, padID int) Packed {
	packed := make([]int, blockSize)
	segID := make([]int, blockSize)
	copy(packed, tokens)
	for i := len(tokens); i < blockSize; i++ {
		packed[i] = padID
	}
	// Assign segment ids.
	pos := 0
	for s, l := range lens {
		for range l {
			segID[pos] = s
			pos++
		}
	}
	for i := pos; i < blockSize; i++ {
		segID[i] = -1
	}
	lensCopy := make([]int, len(lens))
	copy(lensCopy, lens)
	return Packed{Tokens: packed, SegmentLengths: lensCopy, SegmentID: segID}
}

// BlockMask builds the additive block-diagonal causal attention mask for a
// packed block. mask[i][j] is 0 when position i may attend to position j
// (same segment and j ≤ i, neither padding) and a large negative value
// (negInf) otherwise. The result is a blockSize×blockSize float32 mlx.Array
// suitable for adding to attention scores.
//
// This is the mask a packing-aware model forward would consume. The public
// mlx-go-lm forward does not accept it (see package docs); BlockMask exists so
// the mask is correct and tested independent of that gate.
func BlockMask(p Packed) *mlx.Array {
	n := len(p.SegmentID)
	const negInf = float32(-1e9)
	vals := make([]float32, n*n)
	for i := range n {
		for j := range n {
			vals[i*n+j] = negInf
			if p.SegmentID[i] < 0 || p.SegmentID[j] < 0 {
				continue // padding never attends or is attended to
			}
			if p.SegmentID[i] == p.SegmentID[j] && j <= i {
				vals[i*n+j] = 0
			}
		}
	}
	return mlx.NewArray(vals, n, n)
}

// LossMask returns a float32 slice of length BlockSize that is 1 on real tokens
// and 0 on padding, so training ignores padded positions.
func (p Packed) LossMask() []float32 {
	m := make([]float32, len(p.SegmentID))
	for i, s := range p.SegmentID {
		if s >= 0 {
			m[i] = 1
		}
	}
	return m
}

// RealTokens returns the number of non-padding tokens in the block.
func (p Packed) RealTokens() int {
	var n int
	for _, l := range p.SegmentLengths {
		n += l
	}
	return n
}
