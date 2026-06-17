package decontam

import (
	"reflect"
	"testing"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"lowercases", "Hello World", "hello world"},
		{"strips punctuation", "what is 2 + 2? (think!)", "what is 2 2 think"},
		{"collapses whitespace", "a   b\t\nc", "a b c"},
		{"trims ends", "  padded  ", "padded"},
		{"symbols separate tokens", "x = α·β #tag", "x α β tag"},
		{"empty", "   ", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Normalize(tt.in); got != tt.want {
				t.Fatalf("Normalize(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// Invariant (DESIGN §5.7): a train sample containing a verbatim 10-gram from an
// eval sample is DROPPED.
func TestVerbatimTenGramDropped(t *testing.T) {
	eval := []string{"Find the smallest positive integer n such that n squared plus n is divisible by twelve."}
	// The train text embeds a verbatim 10-token (normalized) run from the eval
	// question inside other text.
	train := []string{
		"Solution: find the smallest positive integer n such that n squared plus n is what we seek.",
	}
	// Confirm the shared 10-gram exists at the normalized level (sanity guard
	// so the test asserts real overlap, not an artifact).
	evalGrams, _ := NGrams(eval[0], DefaultN)
	trainGrams, _ := NGrams(train[0], DefaultN)
	shared := false
	for g := range trainGrams {
		if _, ok := evalGrams[g]; ok {
			shared = true
			break
		}
	}
	if !shared {
		t.Fatal("test setup: train and eval share no 10-gram")
	}
	kept, err := Filter(train, eval, DefaultN)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(kept) != 0 {
		t.Fatalf("verbatim 10-gram overlap not dropped: kept %v", kept)
	}
}

// Invariant (DESIGN §5.7): a paraphrase that shares no 10-gram stays below
// threshold and is KEPT.
func TestParaphraseKept(t *testing.T) {
	eval := []string{"Find the smallest positive integer n such that n squared plus n is divisible by twelve."}
	// A paraphrase: same meaning, reworded enough that no run of ten consecutive
	// normalized tokens matches.
	train := []string{"What is the least whole number whose square added to itself can be evenly split by 12?"}

	// Sanity: they may share short n-grams but not a full 10-gram.
	evalGrams, _ := NGrams(eval[0], DefaultN)
	trainGrams, _ := NGrams(train[0], DefaultN)
	for g := range trainGrams {
		if _, ok := evalGrams[g]; ok {
			t.Fatalf("test setup: paraphrase unexpectedly shares 10-gram %q", g)
		}
	}
	kept, err := Filter(train, eval, DefaultN)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(kept) != 1 {
		t.Fatalf("paraphrase should be kept, got %v", kept)
	}
}

// Invariant (DESIGN §5.7): samples shorter than n tokens are handled with no
// spurious overlap.
func TestShortSampleNoSpuriousOverlap(t *testing.T) {
	// Both train and eval are shorter than 10 tokens and even identical; with no
	// 10-gram in either, there is nothing to collide, so the train sample stays.
	eval := []string{"two plus two equals four"}
	train := []string{"two plus two equals four"}

	g, _ := NGrams(eval[0], DefaultN)
	if len(g) != 0 {
		t.Fatalf("short text produced %d 10-grams, want 0", len(g))
	}
	kept, err := Filter(train, eval, DefaultN)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(kept) != 1 {
		t.Fatalf("short identical sample spuriously dropped: kept %v", kept)
	}

	// But at a small n, the identical short texts DO share an n-gram and are
	// dropped — confirming the short-sample behavior is purely the n-gram-count
	// effect, not a special case.
	keptSmallN, _ := Filter(train, eval, 3)
	if len(keptSmallN) != 0 {
		t.Fatalf("identical text should overlap at n=3, kept %v", keptSmallN)
	}
}

func TestNGrams(t *testing.T) {
	got, err := NGrams("a B, a b a", 2)
	if err != nil {
		t.Fatalf("NGrams: %v", err)
	}
	// normalized tokens: [a b a b a] ⇒ bigrams {"a b","b a"}.
	want := map[string]struct{}{"a b": {}, "b a": {}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("NGrams = %v, want %v", got, want)
	}
	// Fewer tokens than n ⇒ empty set.
	empty, _ := NGrams("a b", 5)
	if len(empty) != 0 {
		t.Fatalf("short text n-grams = %v, want empty", empty)
	}
}

// Filter preserves input order of kept train samples and drops only overlapping
// ones.
func TestFilterOrderAndSelectivity(t *testing.T) {
	eval := []string{"alpha beta gamma delta epsilon zeta eta theta iota kappa"}
	train := []string{
		"clean one with totally different words here and there now",        // kept
		"prefix alpha beta gamma delta epsilon zeta eta theta iota kappa",  // dropped (verbatim 10-gram)
		"another clean sample entirely unrelated to the eval question yes", // kept
	}
	kept, err := Filter(train, eval, DefaultN)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(kept) != 2 || kept[0] != train[0] || kept[1] != train[2] {
		t.Fatalf("filter order/selectivity wrong: %v", kept)
	}
}

func TestValidation(t *testing.T) {
	if _, err := NGrams("a b c", 0); err == nil {
		t.Fatal("want error for n=0")
	}
	if _, err := Filter(nil, nil, -1); err == nil {
		t.Fatal("want error for n<0")
	}
}
