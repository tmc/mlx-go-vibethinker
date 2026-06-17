//go:build !modelir

package main

import (
	"fmt"

	"github.com/tmc/mlx-go-vibethinker/ssp"
)

// buildToyPipeline is unavailable without the modelir build tag, which the toy
// model registry requires. Build with -tags modelir to run the toy pipeline.
func buildToyPipeline(size, dir string, seed uint64) (*ssp.Pipeline, error) {
	return nil, fmt.Errorf("the toy pipeline requires the modelir build tag; rebuild with: go build -tags modelir ./cmd/vibethinker-train")
}
