//go:build modelir

package main

import "github.com/tmc/mlx-go-vibethinker/eval/methodcompare"

// evaluateWithModel runs the comparison including the toy-model loss/wall-time.
func evaluateWithModel(seed uint64) ([]methodcompare.Metrics, error) {
	return methodcompare.EvaluateWithModel(seed)
}
