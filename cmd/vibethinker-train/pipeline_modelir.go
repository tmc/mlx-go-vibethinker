//go:build modelir

package main

import (
	"fmt"

	"github.com/tmc/mlx-go-vibethinker/internal/recipe"
	"github.com/tmc/mlx-go-vibethinker/internal/toymodel"
	"github.com/tmc/mlx-go-vibethinker/ssp"
)

// buildToyPipeline constructs the toy recipe for the given size.
func buildToyPipeline(size, dir string, seed uint64) (*ssp.Pipeline, error) {
	lm, err := toymodel.New(toymodel.DefaultConfig(), seed)
	if err != nil {
		return nil, fmt.Errorf("build toy model: %w", err)
	}
	toy := &recipe.Toy{Model: lm, Tok: toymodel.Tokenizer{}, Dir: dir}
	switch size {
	case "1.5b":
		return toy.Pipeline15B(), nil
	case "3b":
		return toy.Pipeline3B(), nil
	default:
		return nil, fmt.Errorf("unknown size %q (want 1.5b or 3b)", size)
	}
}
