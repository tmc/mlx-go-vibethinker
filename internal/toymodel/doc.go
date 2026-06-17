// Package toymodel provides a tiny Qwen2 model and tokenizer used to drive the
// VibeThinker pipeline end to end in tests, without GPU-scale compute.
//
// New builds a 2-layer synthetic Qwen2 with deterministic random weights via
// the mlx-go-lm model registry. Save and Load round-trip its weights through
// safetensors so the merge step (spectrum/fuse) can fuse toy checkpoints, and
// so a stage can persist its output for the next stage. Tokenizer is a
// byte-level tokenizer adequate for exercising the data, packing, and
// log-probability paths.
//
// Nothing here is meant for real training; it exists so the full stage list
// runs and the invariants are exercised on a model small enough to evaluate on
// CPU.
package toymodel
