// Package ssp is the Spectrum-to-Signal orchestration spine for the
// VibeThinker reproduction.
//
// The Spectrum-to-Signal Principle (SSP) structures post-training as an
// ordered list of stages: supervised fine-tuning first builds a diverse
// "spectrum" of solutions (maximizing Pass@K), then reinforcement learning
// amplifies the correct "signal". ssp does not implement any stage itself; it
// defines the contract every stage satisfies and runs them in sequence,
// threading a [Checkpoint] from one stage to the next and recording
// [Provenance] so a finished run is fully auditable.
//
// A [Stage] consumes a checkpoint and produces a new one:
//
//	type Stage interface {
//		Name() string
//		Run(ctx context.Context, in *Checkpoint) (*Checkpoint, error)
//	}
//
// A [Pipeline] is an ordered list of stages. Running it feeds the initial
// checkpoint through each stage in turn; each output checkpoint carries a
// [Provenance] entry naming the stage that produced it and the checkpoint it
// consumed. The 1.5B and 3B recipes are two different pipelines over the same
// stage implementations from the other packages in this module.
//
// Checkpoints are intentionally opaque about model storage: a checkpoint is a
// directory path plus metadata. Stages load and save model weights through the
// mlx-go substrate; ssp only tracks where a checkpoint lives and how it was
// made.
package ssp
