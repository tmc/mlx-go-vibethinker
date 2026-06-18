// Package realmodel is a real-model mechanism smoke test for the post-GRPO
// upgrades in signal/mgpo (DESIGN_RL_UPGRADE.md). Where eval/methodcompare runs
// the methods on a deterministic synthetic scenario to prove the mechanism on a
// toy substrate, this package runs them on a REAL model — Qwen2.5-Math-1.5B —
// with real logits and real optimizer steps.
//
// IMPORTANT: this is a MECHANISM + STABILITY smoke test, NOT a benchmark-accuracy
// reproduction. It validates that each method's mechanism holds on real logits
// (Dr.GRPO shrinks |A|/std on real rollouts; Clip-Higher changes the upper
// clip-bind rate; DCPO-SAS smooths advantage variance across real steps; Dynamic
// Sampling drops real acc∈{0,1} groups; FRPO's future-KL is finite and nonzero;
// HDPO's cliff set is nonempty on real zero-reward groups; DRA's diversity
// reweight moves |A|) and that training stays stable (finite, non-diverging loss)
// over a short run. Reproducing VibeThinker's published accuracy would need
// ~3.9K H800 GPU-hours, a frontier teacher, and the full data/sandbox seams —
// explicitly out of scope.
//
// All numbers here are model- and machine-dependent and NOT bit-reproducible: a
// real model plus a real optimizer is not deterministic across runs the way the
// tag-free toy harness is. The toy harness (eval/methodcompare) remains the
// byte-identical-reproducibility one; this layer is its real-logit companion.
//
// Everything in this package is behind the //go:build modelir tag (the same tag
// the toy model pipeline uses) so the tag-free build, the toy method-comparison
// harness, and the nine signal/mgpo method tests stay green and untouched.
//
// The model weights are not embedded or committed (multi-GB safetensors). The
// loader reads an on-disk HuggingFace directory; point it at one with
// [DefaultModelDir] or the VIBETHINKER_REALMODEL_DIR environment variable.
package realmodel
