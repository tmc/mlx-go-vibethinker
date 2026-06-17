package teacher

import (
	"context"
	"testing"
)

func TestFakeIsDeterministic(t *testing.T) {
	f := Fake{}
	ctx := context.Background()
	a, _, err := f.PseudoLabel(ctx, "what is 2+2", 8)
	if err != nil {
		t.Fatalf("PseudoLabel: %v", err)
	}
	b, _, _ := f.PseudoLabel(ctx, "what is 2+2", 8)
	if a != b {
		t.Fatalf("fake not deterministic: %q vs %q", a, b)
	}
	// Different prompts generally differ.
	c, _, _ := f.PseudoLabel(ctx, "a different prompt entirely", 8)
	if a == c {
		t.Log("hash collision (acceptable, rare)")
	}
}

func TestFakeAnswerOverride(t *testing.T) {
	f := Fake{AnswerFor: map[string]string{"q": "42"}}
	traces, err := f.Traces(context.Background(), "q", 3)
	if err != nil {
		t.Fatalf("Traces: %v", err)
	}
	if len(traces) != 3 {
		t.Fatalf("got %d traces, want 3", len(traces))
	}
	for _, tr := range traces {
		if tr.Answer != "42" {
			t.Fatalf("answer = %q, want override 42", tr.Answer)
		}
	}
}

func TestFakeClaimsAllValid(t *testing.T) {
	f := Fake{}
	claims, err := f.Claims(context.Background(), Trace{Answer: "7"}, 5)
	if err != nil {
		t.Fatalf("Claims: %v", err)
	}
	if len(claims) != 5 {
		t.Fatalf("got %d claims, want 5", len(claims))
	}
	for _, c := range claims {
		if !c.Valid {
			t.Fatal("fake claims should all be valid")
		}
	}
}

func TestFakeRejectsNonPositiveCounts(t *testing.T) {
	f := Fake{}
	ctx := context.Background()
	if _, err := f.Traces(ctx, "q", 0); err == nil {
		t.Fatal("want error for n=0")
	}
	if _, _, err := f.PseudoLabel(ctx, "q", -1); err == nil {
		t.Fatal("want error for n=-1")
	}
	if _, err := f.Claims(ctx, Trace{}, 0); err == nil {
		t.Fatal("want error for m=0")
	}
}

// Fake satisfies the Teacher interface.
var _ Teacher = Fake{}
