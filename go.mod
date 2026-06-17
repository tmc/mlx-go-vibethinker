module github.com/tmc/mlx-go-vibethinker

go 1.26.3

replace (
	github.com/tmc/mlx-go => ../mlx-go
	github.com/tmc/mlx-go-lm => ../mlx-go-lm
	github.com/tmc/mlx-go/examples/mlx-go-distill => ../mlx-go-examples/mlx-go-distill
	github.com/tmc/mlx-go/examples/mlx-go-rl => ../mlx-go-examples/mlx-go-rl
	github.com/tmc/modelir => ../modelir
)

require (
	github.com/tmc/mlx-go v0.0.0
	github.com/tmc/mlx-go-lm v0.0.0
	github.com/tmc/mlx-go/examples/mlx-go-rl v0.0.0-00010101000000-000000000000
)

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/tmc/mlx-go/examples/mlx-go-distill v0.0.0 // indirect
	github.com/tmc/modelir v0.1.2-0.20260517090425-24c01509645e // indirect
	golang.org/x/image v0.39.0 // indirect
	rsc.io/omap v1.2.0 // indirect
)
