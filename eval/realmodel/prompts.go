//go:build modelir

package realmodel

// mathPrompt is one math problem and its gold answer, scored with
// reward/mathverify. The prompts are deliberately small, short-horizon
// arithmetic/algebra so the 1.5B model produces a mix of correct and incorrect
// rollouts under sampling — which is what gives the group-relative advantage a
// non-degenerate spread (and populates HDPO's zero-reward cliff set and Dynamic
// Sampling's drop set) on a real model within a short smoke run.
type mathPrompt struct {
	Question string
	Gold     string
}

// mathPrompts is the fixed dozen-prompt smoke set. The phrasing asks for a
// \boxed{} final answer, which mathverify.ExtractAnswer keys on.
func mathPrompts() []mathPrompt {
	return []mathPrompt{
		{"What is 7 + 8? Give the final answer in \\boxed{}.", "15"},
		{"What is 12 - 5? Give the final answer in \\boxed{}.", "7"},
		{"What is 6 times 4? Give the final answer in \\boxed{}.", "24"},
		{"What is 36 divided by 6? Give the final answer in \\boxed{}.", "6"},
		{"What is 9 + 14? Give the final answer in \\boxed{}.", "23"},
		{"What is 15 - 8? Give the final answer in \\boxed{}.", "7"},
		{"What is 3 times 9? Give the final answer in \\boxed{}.", "27"},
		{"What is half of 50? Give the final answer in \\boxed{}.", "25"},
		{"What is 11 + 19? Give the final answer in \\boxed{}.", "30"},
		{"If x + 4 = 10, what is x? Give the final answer in \\boxed{}.", "6"},
		{"What is 100 - 37? Give the final answer in \\boxed{}.", "63"},
		{"What is 8 squared? Give the final answer in \\boxed{}.", "64"},
	}
}
