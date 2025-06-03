package engine

import (
	"context"

	"github.com/firebase/genkit/go/genkit"
	mygenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/jcooky/go-din"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/habiliai/agentruntime/internal/rag"
	"github.com/habiliai/agentruntime/internal/tool"
)

type (
	Engine interface {
		NewAgentFromConfig(
			ctx context.Context,
			ac config.AgentConfig,
		) (*entity.Agent, error)
		Run(ctx context.Context, req RunRequest, output any) (*RunResponse, error)
		Generate(ctx context.Context, req *GenerateRequest, out any, opts ...ai.GenerateOption) (*ai.ModelResponse, error)
		Embed(ctx context.Context, texts ...string) ([][]float32, error)
		IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error
		RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error)
	}

	engine struct {
		logger      *mylog.Logger
		toolManager tool.Manager
		genkit      *genkit.Genkit
		ragIndexer  rag.Indexer
	}
)

func init() {
	din.RegisterT(func(c *din.Container) (Engine, error) {
		logger := din.MustGet[*mylog.Logger](c, mylog.Key)

		// Create RAG indexer
		embedder := &engineEmbedder{
			genkit: din.MustGet[*genkit.Genkit](c, mygenkit.Key),
		}

		ragIndexer, err := rag.NewIndexer("knowledge.db", embedder)
		if err != nil {
			logger.Error("failed to create RAG indexer", "error", err)
			// Continue without RAG functionality
			ragIndexer = &noopIndexer{}
		}

		return &engine{
			logger:      logger,
			toolManager: din.MustGetT[tool.Manager](c),
			genkit:      din.MustGet[*genkit.Genkit](c, mygenkit.Key),
			ragIndexer:  ragIndexer,
		}, nil
	})
}

// engineEmbedder implements rag.Embedder using the engine's embed functionality
type engineEmbedder struct {
	genkit *genkit.Genkit
}

func (e *engineEmbedder) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	embedder := genkit.LookupEmbedder(e.genkit, "openai", "text-embedding-3-small")

	resp, err := ai.Embed(ctx, embedder, ai.WithTextDocs(texts...))
	if err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(resp.Embeddings))
	for i, embedding := range resp.Embeddings {
		embeddings[i] = embedding.Embedding
	}

	return embeddings, nil
}

// noopIndexer is a no-op implementation for when RAG indexer fails to initialize
type noopIndexer struct{}

func (n *noopIndexer) IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error {
	return nil
}

func (n *noopIndexer) RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error) {
	return nil, nil
}

func (n *noopIndexer) DeleteAgentKnowledge(ctx context.Context, agentName string) error {
	return nil
}

func (n *noopIndexer) Close() error {
	return nil
}

// IndexKnowledge indexes knowledge documents for an agent
func (e *engine) IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error {
	return e.ragIndexer.IndexKnowledge(ctx, agentName, knowledge)
}

// RetrieveRelevantKnowledge retrieves relevant knowledge chunks based on query
func (e *engine) RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error) {
	return e.ragIndexer.RetrieveRelevantKnowledge(ctx, agentName, query, limit)
}
