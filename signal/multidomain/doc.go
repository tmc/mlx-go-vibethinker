// Package multidomain drives VibeThinker's 3B multi-domain RL: a sequence of
// MGPO runs over Math, then Code, then STEM (DESIGN §4.2).
//
// Each domain is a separate MGPO run that starts from the previous domain's
// output checkpoint and uses that domain's reward source (answer equivalence
// for math, sandbox execution for code, answer-plus-option for STEM). The
// checkpoint produced after each domain is retained, because Offline
// Self-Distillation later rejection-samples from all three.
//
// [Run] threads the checkpoint through the ordered domains via an injected
// [DomainRunner] seam — the actual MGPO optimization lives behind that seam so
// the ordering, per-domain reward routing, and checkpoint retention are tested
// without a GPU. The order and the retained-checkpoint set are the properties
// the tests pin.
package multidomain
