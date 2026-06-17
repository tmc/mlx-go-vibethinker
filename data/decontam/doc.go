// Package decontam removes train samples that overlap an evaluation set by a
// shared 10-gram, the contamination test VibeThinker applies before training
// (DESIGN §4.6, §5.7).
//
// Each sample's text is normalized — lowercased, punctuation and symbols
// stripped, runs of whitespace collapsed — and split into tokens. The set of
// contiguous token n-grams (default n = 10) of every eval sample is collected
// into one eval n-gram set. A train sample is dropped iff it shares any n-gram
// with that set:
//
//	drop(train) ⇔ ngrams_n(train) ∩ ⋃_e ngrams_n(eval_e) ≠ ∅.
//
// The threshold is exact-match on a single n-gram, so a verbatim 10-gram copied
// from an eval question is always caught, while a paraphrase that never repeats
// ten consecutive normalized tokens shares no n-gram and is kept. A text with
// fewer than n tokens has no n-grams and therefore cannot collide — short
// samples never produce a spurious overlap. These three behaviors (verbatim
// dropped, paraphrase kept, short handled) are pinned by the tests.
//
// [Normalize] exposes the text normalization, [NGrams] the n-gram extraction,
// and [Filter] the end-to-end train-vs-eval decontamination, each a thin shell
// over an unexported core.
package decontam
