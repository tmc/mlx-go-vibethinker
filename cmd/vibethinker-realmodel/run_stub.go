//go:build !modelir

package main

import "fmt"

// run is unavailable without the modelir build tag, which the real-model loader
// requires. Rebuild with -tags modelir.
func run(o opts) error {
	return fmt.Errorf("vibethinker-realmodel requires the modelir build tag; rebuild with: go run -tags modelir ./cmd/vibethinker-realmodel")
}
