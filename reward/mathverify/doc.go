// Package mathverify implements VibeThinker's rule-based final-answer
// equivalence reward (DESIGN §4.5: "mathverify — rule-based final-answer
// equivalence (local)").
//
// The reward is binary: a completion scores 1 when its extracted final answer
// is equivalent to the reference answer, and 0 otherwise. This is the
// verifiable reward the MGPO/GRPO optimizer consumes for math prompts —
//
//	r(q, y) = 1 if answer(y) ≡ answer(a), else 0,
//
// where answer(·) extracts the final answer (the last \boxed{...} if present,
// otherwise the trailing number or expression) and ≡ is value equivalence
// after normalization: surrounding $ and \text wrappers stripped, whitespace
// and thousands-commas removed, trailing zeros and a trailing decimal point
// trimmed, and simple fraction a/b reduced to its decimal value. So "2/4" ≡
// "0.5", "12" ≡ "12.0", and "$1{,}000$" ≡ "1000"; a missing or non-equivalent
// answer scores 0. Extraction and equivalence are deterministic and local —
// no model call — which is what makes this reward verifiable. The 3B recipe
// additionally pairs this with an LLM-as-judge for complex forms; that judge
// is a separate gated seam and is not implemented here.
//
// Verify and Environment adapt the reward to mlx-go-rl's RichEnvironment so it
// composes with the rest of the reward pipeline. The package-level Equivalent,
// ExtractAnswer, and Normalize entrypoints are thin shells over the unexported
// numeric/parsing cores the tests target.
package mathverify
