// Package sft drives VibeThinker's curriculum supervised fine-tuning and
// provides the sequence packing that the mlx-go-lm training loop does not.
//
// # Sequence packing
//
// mlx-go-lm ships only padding batch iterators; it has no sequence packing.
// Packing concatenates several short token sequences into one fixed-length
// block so no compute is wasted on padding. To keep packed sequences from
// attending across their boundaries, packing also builds the block-diagonal
// attention mask and per-position segment ids: position t may attend to
// position s only when s ≤ t and both lie in the same segment. [Pack] returns
// the packed token blocks together with that mask and the segment lengths, all
// fully tested in isolation.
//
// The public mlx-go-lm model Forward(ctx, inputs, cache) builds its causal mask
// internally and does not accept an external attention mask, so consuming the
// block-diagonal mask inside attention is a substrate gate: [PackedTrainer]
// documents exactly where a mask-aware forward plugs in. The packing transform
// and its mask are real and verified regardless.
//
// # Curriculum
//
// The 3B recipe runs two SFT stages (DESIGN §4.1): stage 1 is broad-coverage
// cold-start over the full quality-filtered set; stage 2 fine-tunes the stage-1
// checkpoint on a hard-reasoning subset, keeping only problems whose reference
// rollouts fail often enough (drop traces shorter than a token floor; drop
// problems with rollout error rate below a threshold). [Curriculum] expresses
// the two stages and the hard-subset [Filter] as plain data so the recipe is
// declarative; the actual optimization is delegated to an injected [Trainer]
// seam, which a real run backs with mlx-go-lm's full fine-tune.
package sft
