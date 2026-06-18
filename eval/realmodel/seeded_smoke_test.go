//go:build modelir

package realmodel

import (
	"context"
	"testing"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
)

// The seeded source must yield genuine within-group reward spread (mixed
// correctness), so Dr.GRPO's std-removal shows up as a different |A| than
// baseline on the SAME seeded reward vectors. One model is loaded; the base
// policy is restored from a snapshot between methods (reloading a second model
// OOMs the wired working set).
func TestSeededGivesMixedCorrectnessAndDrGRPODelta(t *testing.T) {
	m := requireModel(t)
	ctx := context.Background()
	cfg := smokeConfig()

	slots, snap, err := snapshotParams(m)
	if err != nil {
		t.Fatalf("snapshotParams: %v", err)
	}

	reg := methodcompare.Registry()
	pick := func(name string) Method {
		for _, mm := range reg {
			if mm.Name == name {
				return mm
			}
		}
		t.Fatalf("method %q not found", name)
		return Method{}
	}

	run := func(method Method, restore bool) Metrics {
		if restore {
			if err := restoreParams(slots, snap); err != nil {
				t.Fatalf("restoreParams: %v", err)
			}
		}
		g, err := buildSeededGroups(ctx, m, cfg)
		if err != nil {
			t.Fatalf("buildSeededGroups: %v", err)
		}
		mt, err := runMethod(ctx, m, method, cfg, g)
		if err != nil {
			t.Fatalf("runMethod %s: %v", method.Name, err)
		}
		return mt
	}

	mtB := run(pick("baseline"), false)
	mtD := run(pick("+DrGRPO"), true)

	t.Logf("SEEDED baseline: acc=%.2f learn=%d cliff=%d advStd=%.4f advAbsMean=%.4f ratioVar=%.3g",
		mtB.AccMean, mtB.LearnGroups, mtB.CliffGroups, mtB.AdvStd, mtB.AdvAbsMean, mtB.RatioVar)
	t.Logf("SEEDED +DrGRPO:  acc=%.2f learn=%d cliff=%d advStd=%.4f advAbsMean=%.4f ratioVar=%.3g",
		mtD.AccMean, mtD.LearnGroups, mtD.CliffGroups, mtD.AdvStd, mtD.AdvAbsMean, mtD.RatioVar)

	if mtB.LearnGroups == 0 {
		t.Errorf("seeded baseline has no learnable (mixed-correctness) groups; spread not exhibited")
	}
	if mtB.AdvAbsMean == mtD.AdvAbsMean {
		t.Errorf("Dr.GRPO advAbsMean == baseline (%v); std-removal not visible on seeded spread", mtB.AdvAbsMean)
	}
}
