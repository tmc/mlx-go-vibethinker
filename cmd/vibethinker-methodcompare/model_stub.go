//go:build !modelir

package main

import (
	"fmt"

	"github.com/tmc/mlx-go-vibethinker/eval/methodcompare"
)

// evaluateWithModel is unavailable without the modelir build tag, which the toy
// model registry requires. Rebuild with -tags modelir to use -model.
func evaluateWithModel(seed uint64) ([]methodcompare.Metrics, error) {
	return nil, fmt.Errorf("-model requires the modelir build tag; rebuild with: go build -tags modelir ./cmd/vibethinker-methodcompare")
}
