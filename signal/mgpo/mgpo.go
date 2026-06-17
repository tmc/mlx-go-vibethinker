package mgpo

import (
	"fmt"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
	"github.com/tmc/mlx-go/mlx"
)

// ScaledAdvantages computes MGPO's modulated advantages from per-group rewards.
//
// rewards[i] is the reward vector of group i (one entry per rollout). For each
// group it computes the normalized GRPO advantage with rl.GroupAdvantage, the
// empirical accuracy p_c, and the weight w_ME(p_c) = exp(−λ·D_ME), then returns
// w_ME·A per group. Because GroupAdvantage normalizes by the group std, scaling
// the advantage (not the reward) is the only placement that does not cancel,
// and it matches the paper's A′ⱼ(q) = w_ME(p_c(q))·Aⱼ(q).
//
// At λ = 0, every w_ME is 1, so the result equals rl.GroupAdvantage(rewards)
// exactly — MGPO degenerates to GRPO. lambda must be ≥ 0.
func ScaledAdvantages(rewards [][]float64, lambda float64) ([][]float64, error) {
	if lambda < 0 {
		return nil, fmt.Errorf("mgpo: lambda must be >= 0, got %v", lambda)
	}
	return scaledAdvantages(rl.GroupAdvantage(rewards), rewards, lambda)
}

// scaledAdvantages applies the per-group weight to precomputed advantages. adv
// and rewards must have the same group/rollout shape. It is the unexported core
// that the tests target directly.
func scaledAdvantages(adv, rewards [][]float64, lambda float64) ([][]float64, error) {
	if len(adv) != len(rewards) {
		return nil, fmt.Errorf("mgpo: %d advantage groups but %d reward groups", len(adv), len(rewards))
	}
	out := make([][]float64, len(adv))
	for i := range adv {
		if len(adv[i]) != len(rewards[i]) {
			return nil, fmt.Errorf("mgpo: group %d has %d advantages but %d rewards", i, len(adv[i]), len(rewards[i]))
		}
		pc := Accuracy(rewards[i])
		w := weight(lambda, pc)
		scaled := make([]float64, len(adv[i]))
		for j, a := range adv[i] {
			scaled[j] = w * a
		}
		out[i] = scaled
	}
	return out, nil
}

// FlattenAdvantages concatenates per-group advantages into a single
// per-sequence slice in group-major order, matching the order rollouts are laid
// out for rl.GRPOLoss (group 0's rollouts, then group 1's, ...).
func FlattenAdvantages(adv [][]float64) []float64 {
	var n int
	for _, g := range adv {
		n += len(g)
	}
	out := make([]float64, 0, n)
	for _, g := range adv {
		out = append(out, g...)
	}
	return out
}

// AdvantageTensor materializes a per-sequence advantage slice into the
// [numSeq, 1] float32 mlx.Array that rl.GRPOLoss expects (it broadcasts the
// per-sequence advantage across tokens). It returns an error for an empty
// slice.
func AdvantageTensor(adv []float64) (*mlx.Array, error) {
	if len(adv) == 0 {
		return nil, fmt.Errorf("mgpo: empty advantage slice")
	}
	vals := make([]float32, len(adv))
	for i, a := range adv {
		vals[i] = float32(a)
	}
	return mlx.NewArray(vals, len(vals), 1), nil
}

// Loss is a thin convenience wrapper: it scales the rewards' advantages by w_ME,
// flattens and materializes them, and calls the package-level rl.GRPOLoss with
// the given per-token log-prob tensors. current, old, ref, and mask are the
// same tensors rl.GRPOLoss documents (old and ref must already be wrapped in
// mlx.StopGradient). The flattened advantage order must match the sequence order
// of those tensors (group-major).
func Loss(current, old, ref, mask *mlx.Array, rewards [][]float64, lambda float64, config rl.GRPOConfig) (*mlx.Array, error) {
	scaled, err := ScaledAdvantages(rewards, lambda)
	if err != nil {
		return nil, err
	}
	advArr, err := AdvantageTensor(FlattenAdvantages(scaled))
	if err != nil {
		return nil, err
	}
	loss, err := rl.GRPOLoss(current, old, ref, advArr, mask, config)
	if err != nil {
		return nil, fmt.Errorf("mgpo: GRPOLoss: %w", err)
	}
	return loss, nil
}
