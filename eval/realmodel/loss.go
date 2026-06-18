//go:build modelir

package realmodel

import (
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"

	"github.com/tmc/mlx-go-vibethinker/signal/mgpo"
)

// grpoLossDCPO computes the method's MGPO/GRPO scalar loss for one rollout group
// against frozen old/ref snapshots, threading the DCPO running-stats store when
// the method enables smoothing.
//
// When the method uses FRPO it dispatches to the FRPO sibling loss (which has no
// DCPO seam — FRPO and DCPO are independent knobs; the all-on row applies FRPO's
// future-KL and leaves DCPO to the advantage path of the non-FRPO methods). For
// every other method it builds the (optionally DCPO-smoothed) scaled advantages
// and calls the substrate rl.GRPOLoss directly, exactly mirroring mgpo.LossOpt's
// internals but with the stats-threaded advantage seam.
func grpoLossDCPO(current, old, ref, mask *mlx.Array, rewards [][]float64, method Method, cfg rl.GRPOConfig, stats *mgpo.PromptStats, promptIDs []string) (*mlx.Array, error) {
	if method.FRPO.BetaFuture != 0 {
		return mgpo.LossFRPOScaled(current, old, ref, mask, rewards, method.Lambda, cfg, method.Opts, method.FRPO)
	}
	if stats == nil {
		// No DCPO: the plain Tier-1 MGPO loss (bit-identical to mgpo.LossOpt).
		return mgpo.LossOpt(current, old, ref, mask, rewards, method.Lambda, cfg, method.Opts)
	}
	// DCPO smoothing: scaled advantages threaded through the running-stats store,
	// then the substrate surrogate (mirrors mgpo.LossOpt's body).
	scaled, err := mgpo.ScaledAdvantagesStep(rewards, method.Lambda, method.Opts, stats, promptIDs)
	if err != nil {
		return nil, err
	}
	advArr, err := mgpo.AdvantageTensor(mgpo.FlattenAdvantages(scaled))
	if err != nil {
		return nil, err
	}
	// Apply the method's effective clip range to the config (the exported
	// equivalent of the unexported Options.applyClip that LossOpt uses).
	low, high := method.Opts.ClipRange(cfg)
	clipped := cfg
	clipped.ClipEpsLow = low
	clipped.ClipEpsHigh = high
	loss, err := rl.GRPOLoss(current, old, ref, advArr, mask, clipped)
	if err != nil {
		return nil, fmt.Errorf("realmodel: GRPOLoss (DCPO): %w", err)
	}
	return loss, nil
}
