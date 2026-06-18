//go:build modelir

package realmodel

// HeldoutSet is the FIXED held-out probe set used to measure greedy Avg@1
// correctness at step 0 and at the final step of each swept config. It is held
// constant across every config and seed so the only thing moving the per-config
// Δacc is the optimizer steps, never the probe set.
//
// IMPORTANT — this is a DIRECTIONAL signal, not a benchmark:
//   - It is tiny (N=12) and hand-built, scored by greedy Avg@1 against the boxed
//     gold answer with reward/mathverify. It cannot estimate true benchmark
//     accuracy; it estimates the DIRECTION a few real optimizer steps move a
//     fixed policy on a fixed probe.
//   - It is DISJOINT from mathPrompts() (the training prompts): no prompt and no
//     numeric instance is shared, so a measured Δacc is generalization on unseen
//     problems within the same short-horizon arithmetic/algebra distribution.
//   - The difficulty is graded ON PURPOSE so the base Qwen2.5-Math-1.5B sits
//     meaningfully OFF BOTH FLOORS at step 0 (some right, some wrong under greedy
//     decode), giving Δacc room to move. The probe test (heldout_probe_test.go)
//     is the CKPT-A gate that verifies this.
//
// Size and composition are bounded by a hard substrate limit, not by preference:
// the incremental KV cache produces a degenerate repetition loop on this model
// (see generateGreedy), so the held-out decode must use the no-cache re-forward,
// whose live Metal arrays are NOT reclaimable in-process and trip the device
// array ceiling (~499000) at roughly ~1300 cumulative decoded tokens. N=12 with
// the \boxed{} early-stop keeps a full step-0+final pair of passes inside one
// config's subprocess under that ceiling. The set is half "solves-and-boxes-
// early" (low token cost, headroom DOWN) and half "hard-for-a-1.5B" (headroom
// UP), so step-0 baseline lands near the middle.
func HeldoutSet() []mathPrompt {
	return []mathPrompt{
		// --- Solves and boxes early (base reliably right; low token cost). ---
		{"What is 5 + 6? Give the final answer in \\boxed{}.", "11"},
		{"What is 13 - 4? Give the final answer in \\boxed{}.", "9"},
		{"What is 7 times 3? Give the final answer in \\boxed{}.", "21"},
		{"What is 9 squared? Give the final answer in \\boxed{}.", "81"},
		{"What is one third of 27? Give the final answer in \\boxed{}.", "9"},
		{"What is 8 plus 16? Give the final answer in \\boxed{}.", "24"},

		// --- Harder for a 1.5B at a short greedy decode (headroom UP). ---
		{"What is 17 times 6? Give the final answer in \\boxed{}.", "102"},
		{"What is the greatest common divisor of 48 and 36? Give the final answer in \\boxed{}.", "12"},
		{"A shirt costs $40 and is discounted by 15%. What is the sale price in dollars? Give the final answer in \\boxed{}.", "34"},
		{"What is the least common multiple of 4 and 6? Give the final answer in \\boxed{}.", "12"},
		{"What is 256 - 178? Give the final answer in \\boxed{}.", "78"},
		{"If 3x = 51, what is x? Give the final answer in \\boxed{}.", "17"},
	}
}
