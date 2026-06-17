// Package fuse implements Expert Model Fusion: the Expert Model Fusion step of
// VibeThinker's Diversity-Exploring Distillation.
//
// After Domain-Aware Diversity Probing selects a specialist checkpoint per
// subdomain, fusion merges those specialists into a single supervised
// fine-tuning model by a weighted linear average of their parameters:
//
//	M_merge = Σ wᵢ·Mᵢ,  wᵢ ≥ 0,  Σ wᵢ = 1.
//
// The paper's default is a uniform average, wᵢ = 1/N (see [UniformWeights]).
// All specialists must share the same architecture: identical tensor names,
// shapes, and dtypes. Fusion validates that agreement and fails closed on any
// mismatch — silently dropping or zero-filling a missing tensor would corrupt
// the merged model.
//
// [Merge] is a thin validating shell over an unexported averaging core. It
// loads each model's safetensors, checks weight normalization and tensor
// agreement, computes the per-tensor weighted average in float32 for numerical
// stability, casts each result back to the source dtype, and writes the merged
// safetensors to a directory.
//
// Properties that fusion guarantees (and that the tests pin):
//
//   - Uniform merge of N identical models is the identity.
//   - Weights summing to 1 preserve parameter magnitude (the merge of inputs
//     bounded by m is bounded by m).
//   - A name, shape, or dtype mismatch is an error, never a silent fill.
package fuse
