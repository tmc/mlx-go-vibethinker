// Package distill implements VibeThinker's Offline Self-Distillation for the 3B
// model.
//
// From the retained Math/Code/STEM RL checkpoints, traces are rejection-sampled
// with domain verifiers (incorrect ones dropped). For each verified trace y of
// prompt q, the learning-potential score is the length-normalized negative
// log-likelihood of y under the current student π_stu:
//
//	S_LP(q, y) = −(1/|y|) Σ_t log π_stu(y_t | q, y_{<t}).
//
// A higher S_LP means the student models the trace less well, so the trace
// carries more distillation value. [Score] computes S_LP from per-token student
// log-probabilities; [StudentScorer] is the seam that produces those log-probs
// (backed by rl.LogProbs in a real run, by a fake in tests). The ranking
// property — a deliberately mispredicted trace scores above a well-modeled one —
// is pinned by the tests.
//
// Selection is per domain, not global (DESIGN §4.3). Within each domain's own
// length buckets, [Select] drops extremely short traces and high-score outliers
// and keeps the middle-to-high range, then mixes across domains. The student is
// then SFT-ed on the selected set. Bucketing per domain prevents a domain whose
// traces are uniformly long (or short) from dominating a single global scale.
package distill
