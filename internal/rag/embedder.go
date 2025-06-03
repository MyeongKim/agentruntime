package rag

import (
	"context"
)

// EngineEmbedder wraps an engine's Embed method to implement the Embedder interface
type EngineEmbedder struct {
	embedFunc func(ctx context.Context, texts ...string) ([][]float32, error)
}

// NewEngineEmbedder creates a new embedder that uses the provided embed function
func NewEngineEmbedder(embedFunc func(ctx context.Context, texts ...string) ([][]float32, error)) Embedder {
	return &EngineEmbedder{
		embedFunc: embedFunc,
	}
}

// Embed generates embeddings for the given texts
func (e *EngineEmbedder) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	return e.embedFunc(ctx, texts...)
}
