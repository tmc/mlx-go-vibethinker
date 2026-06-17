package data

import (
	"context"
	"testing"
)

// Strata are length-ordered, partition all samples, and are roughly equal-sized.
func TestStratifyByLength(t *testing.T) {
	samples := []Sample{
		{Prompt: "e", Length: 50},
		{Prompt: "a", Length: 10},
		{Prompt: "c", Length: 30},
		{Prompt: "b", Length: 20},
		{Prompt: "d", Length: 40},
		{Prompt: "f", Length: 60},
	}
	strata, err := StratifyByLength(samples, 3)
	if err != nil {
		t.Fatalf("StratifyByLength: %v", err)
	}
	if len(strata) != 3 {
		t.Fatalf("got %d strata, want 3", len(strata))
	}
	// Every sample accounted for exactly once.
	total := 0
	for _, s := range strata {
		total += len(s.Samples)
	}
	if total != len(samples) {
		t.Fatalf("strata hold %d samples, want %d", total, len(samples))
	}
	// Equal-count partition: 6 samples / 3 buckets = 2 each.
	for i, s := range strata {
		if len(s.Samples) != 2 {
			t.Fatalf("stratum %d holds %d, want 2", i, len(s.Samples))
		}
	}
	// Length-ordered: each stratum's max <= next stratum's min.
	for i := 0; i+1 < len(strata); i++ {
		if strata[i].MaxLength > strata[i+1].MinLength {
			t.Fatalf("strata not length-ordered at %d: %d > %d", i, strata[i].MaxLength, strata[i+1].MinLength)
		}
	}
	// Shortest bucket holds the two shortest samples.
	if strata[0].MinLength != 10 || strata[0].MaxLength != 20 {
		t.Fatalf("first stratum bounds = [%d,%d], want [10,20]", strata[0].MinLength, strata[0].MaxLength)
	}
}

// Stratification is independent of input order (stable sort + deterministic
// tiebreak).
func TestStratifyDeterministic(t *testing.T) {
	a := []Sample{{Prompt: "x", Length: 5}, {Prompt: "y", Length: 5}, {Prompt: "z", Length: 5}}
	b := []Sample{{Prompt: "z", Length: 5}, {Prompt: "x", Length: 5}, {Prompt: "y", Length: 5}}
	sa, _ := StratifyByLength(a, 3)
	sb, _ := StratifyByLength(b, 3)
	for i := range sa {
		if len(sa[i].Samples) != len(sb[i].Samples) {
			t.Fatalf("stratum %d sizes differ", i)
		}
		for j := range sa[i].Samples {
			if sa[i].Samples[j].Prompt != sb[i].Samples[j].Prompt {
				t.Fatalf("stratum %d sample %d differs: %q vs %q", i, j, sa[i].Samples[j].Prompt, sb[i].Samples[j].Prompt)
			}
		}
	}
}

func TestStratifyFewerSamplesThanBuckets(t *testing.T) {
	samples := []Sample{{Prompt: "a", Length: 1}, {Prompt: "b", Length: 2}}
	strata, err := StratifyByLength(samples, 5)
	if err != nil {
		t.Fatalf("StratifyByLength: %v", err)
	}
	if len(strata) != 5 {
		t.Fatalf("got %d strata, want 5", len(strata))
	}
	total := 0
	for _, s := range strata {
		total += len(s.Samples)
	}
	if total != 2 {
		t.Fatalf("strata hold %d samples, want 2", total)
	}
}

func TestStratifyValidation(t *testing.T) {
	if _, err := StratifyByLength(nil, 0); err == nil {
		t.Fatal("want error for buckets=0")
	}
}

func TestSliceLoader(t *testing.T) {
	in := []Sample{{Prompt: "q1"}, {Prompt: "q2"}}
	l := SliceLoader{Samples: in}
	out, err := l.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("loaded %d, want 2", len(out))
	}
	// Load returns a copy: mutating the result must not affect the loader.
	out[0].Prompt = "mutated"
	again, _ := l.Load(context.Background())
	if again[0].Prompt != "q1" {
		t.Fatalf("Load did not return a copy: %q", again[0].Prompt)
	}
}

func TestEchoSynthesizer(t *testing.T) {
	seeds := []Sample{{Prompt: "seed", Answer: "a", Domain: "math"}}
	s := EchoSynthesizer{Copies: 3}
	out, err := s.Synthesize(context.Background(), seeds)
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d variants, want 3", len(out))
	}
	for _, v := range out {
		if v.Answer != "a" || v.Domain != "math" {
			t.Fatalf("variant lost seed fields: %+v", v)
		}
		if v.Prompt == "seed" {
			t.Fatal("variant prompt should be marked, got unchanged seed")
		}
	}
	// Non-positive Copies defaults to 1.
	one, _ := EchoSynthesizer{}.Synthesize(context.Background(), seeds)
	if len(one) != 1 {
		t.Fatalf("default copies = %d, want 1", len(one))
	}
}

// The fakes satisfy the seams.
var (
	_ Loader      = SliceLoader{}
	_ Synthesizer = EchoSynthesizer{}
)
