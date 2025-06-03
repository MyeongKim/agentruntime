package rag

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// mockEmbedder for testing
type mockEmbedder struct{}

func (m *mockEmbedder) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		// Create simple mock embeddings (in real usage, these would be from OpenAI)
		embedding := make([]float32, 1536)
		for j := range embedding {
			embedding[j] = float32(i+j) * 0.001 // Simple pattern for testing
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

func TestRAGIndexer(t *testing.T) {
	// Skip test if sqlite-vec is not available
	t.Skip("Skipping RAG test - requires sqlite-vec extension to be installed")

	// Create temporary database
	dbPath := "test_knowledge.db"
	defer os.Remove(dbPath)

	embedder := &mockEmbedder{}
	indexer, err := NewIndexer(dbPath, embedder)
	require.NoError(t, err)
	defer indexer.Close()

	ctx := context.Background()
	agentName := "test-agent"

	// Test knowledge data
	knowledge := []map[string]any{
		{
			"cityName": "Seoul",
			"aliases":  "Seoul, SEOUL, KOR, Korea",
			"info":     "Capital city of South Korea",
		},
		{
			"cityName": "Tokyo",
			"aliases":  "Tokyo, TYO, Japan",
			"info":     "Capital city of Japan",
		},
		{
			"cityName": "New York",
			"aliases":  "NYC, New York City",
			"info":     "Largest city in the United States",
		},
	}

	// Test indexing knowledge
	err = indexer.IndexKnowledge(ctx, agentName, knowledge)
	require.NoError(t, err)

	// Test retrieving relevant knowledge
	results, err := indexer.RetrieveRelevantKnowledge(ctx, agentName, "Korea capital", 2)
	require.NoError(t, err)
	require.NotEmpty(t, results)

	// Test deleting agent knowledge
	err = indexer.DeleteAgentKnowledge(ctx, agentName)
	require.NoError(t, err)

	// Verify knowledge is deleted
	results, err = indexer.RetrieveRelevantKnowledge(ctx, agentName, "Korea capital", 2)
	require.NoError(t, err)
	require.Empty(t, results)
}

func TestKnowledgeProcessing(t *testing.T) {
	embedder := &mockEmbedder{}
	indexer := &sqliteVecIndexer{embedder: embedder}

	knowledge := []map[string]any{
		{
			"cityName": "Seoul",
			"aliases":  "Seoul, SEOUL, KOR, Korea",
		},
		{
			"title":       "Weather Info",
			"description": "Current weather conditions",
		},
		{
			"content": "This is some content",
		},
		{
			"randomField": "Some random data",
		},
	}

	chunks := indexer.processKnowledge(knowledge)
	require.Len(t, chunks, 4)

	// Verify text extraction
	require.Contains(t, chunks[0].Content, "Seoul")
	require.Contains(t, chunks[1].Content, "Weather Info")
	require.Contains(t, chunks[2].Content, "This is some content")
	require.Contains(t, chunks[3].Content, "randomField: Some random data")
}
