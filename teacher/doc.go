// Package teacher defines the Teacher gate: the strong-model dependency that
// VibeThinker's Diversity-Exploring Distillation and pseudo-labeling require but
// that this reproduction cannot bundle.
//
// A real VibeThinker run draws multi-path reasoning traces, majority-vote
// pseudo-labels, and (for the 3B's CLR) decision-relevant claim extraction and
// self-verification from a frontier teacher model. None of that runs locally:
// it needs an API-backed or large local model. So teacher is a pluggable
// interface with an in-repo deterministic fake for tests, not a hidden network
// call.
//
// The gate is explicit. [Teacher] is the seam; [Fake] satisfies it for tests
// and the toy pipeline. A production run supplies a real implementation. The
// exact command and reason a real teacher is not exercised here are recorded at
// the call sites and in the run report.
package teacher
