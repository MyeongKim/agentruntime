package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/habiliai/agentruntime/errors"
	_ "github.com/mattn/go-sqlite3"
)

type (
	// Indexer handles vector indexing and retrieval for RAG
	Indexer interface {
		// IndexKnowledge indexes knowledge documents for an agent
		IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error

		// RetrieveRelevantKnowledge retrieves relevant knowledge chunks based on query
		RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error)

		// DeleteAgentKnowledge removes all knowledge for an agent
		DeleteAgentKnowledge(ctx context.Context, agentName string) error

		// Close closes the indexer
		Close() error
	}

	// KnowledgeChunk represents a piece of indexed knowledge
	KnowledgeChunk struct {
		ID        int64   `json:"id"`
		AgentName string  `json:"agent_name"`
		Content   string  `json:"content"`
		Metadata  string  `json:"metadata"`
		Distance  float64 `json:"distance,omitempty"`
	}

	// Embedder interface for generating embeddings
	Embedder interface {
		Embed(ctx context.Context, texts ...string) ([][]float32, error)
	}

	// sqliteVecIndexer implements Indexer using sqlite-vec
	sqliteVecIndexer struct {
		db       *sql.DB
		embedder Embedder
	}
)

// NewIndexer creates a new RAG indexer using sqlite-vec
func NewIndexer(dbPath string, embedder Embedder) (Indexer, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open sqlite database")
	}

	indexer := &sqliteVecIndexer{
		db:       db,
		embedder: embedder,
	}

	if err := indexer.initialize(); err != nil {
		db.Close()
		return nil, errors.Wrapf(err, "failed to initialize indexer")
	}

	return indexer, nil
}

// initialize sets up the sqlite-vec extension and creates necessary tables
func (idx *sqliteVecIndexer) initialize() error {
	// Load sqlite-vec extension
	if _, err := idx.db.Exec("SELECT load_extension('vec0')"); err != nil {
		return errors.Wrapf(err, "failed to load sqlite-vec extension")
	}

	// Create knowledge table with vector embeddings
	createTableSQL := `
		CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vectors USING vec0(
			agent_name TEXT,
			content TEXT,
			metadata TEXT,
			embedding float[1536]
		);
		
		CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge_vectors(agent_name);
	`

	if _, err := idx.db.Exec(createTableSQL); err != nil {
		return errors.Wrapf(err, "failed to create knowledge_vectors table")
	}

	return nil
}

// IndexKnowledge indexes knowledge documents for an agent
func (idx *sqliteVecIndexer) IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error {
	// First, delete existing knowledge for this agent
	if err := idx.DeleteAgentKnowledge(ctx, agentName); err != nil {
		return errors.Wrapf(err, "failed to delete existing knowledge")
	}

	if len(knowledge) == 0 {
		return nil
	}

	// Process knowledge into text chunks
	chunks := idx.processKnowledge(knowledge)
	if len(chunks) == 0 {
		return nil
	}

	// Extract text content for embedding
	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Content
	}

	// Generate embeddings
	embeddings, err := idx.embedder.Embed(ctx, texts...)
	if err != nil {
		return errors.Wrapf(err, "failed to generate embeddings")
	}

	if len(embeddings) != len(chunks) {
		return errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(chunks))
	}

	// Insert chunks with embeddings
	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrapf(err, "failed to begin transaction")
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO knowledge_vectors (agent_name, content, metadata, embedding)
		VALUES (?, ?, ?, ?)
	`)
	if err != nil {
		return errors.Wrapf(err, "failed to prepare insert statement")
	}
	defer stmt.Close()

	for i, chunk := range chunks {
		embeddingJSON, err := json.Marshal(embeddings[i])
		if err != nil {
			return errors.Wrapf(err, "failed to marshal embedding")
		}

		if _, err := stmt.Exec(agentName, chunk.Content, chunk.Metadata, string(embeddingJSON)); err != nil {
			return errors.Wrapf(err, "failed to insert knowledge chunk")
		}
	}

	if err := tx.Commit(); err != nil {
		return errors.Wrapf(err, "failed to commit transaction")
	}

	return nil
}

// RetrieveRelevantKnowledge retrieves relevant knowledge chunks based on query
func (idx *sqliteVecIndexer) RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error) {
	// Generate embedding for the query
	embeddings, err := idx.embedder.Embed(ctx, query)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate query embedding")
	}

	if len(embeddings) == 0 {
		return nil, errors.Errorf("no embedding generated for query")
	}

	queryEmbedding := embeddings[0]
	embeddingJSON, err := json.Marshal(queryEmbedding)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal query embedding")
	}

	// Perform vector similarity search
	searchSQL := `
		SELECT content, distance
		FROM knowledge_vectors
		WHERE agent_name = ? AND embedding MATCH ?
		ORDER BY distance
		LIMIT ?
	`

	rows, err := idx.db.QueryContext(ctx, searchSQL, agentName, string(embeddingJSON), limit)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to execute similarity search")
	}
	defer rows.Close()

	var results []string
	for rows.Next() {
		var content string
		var distance float64
		if err := rows.Scan(&content, &distance); err != nil {
			return nil, errors.Wrapf(err, "failed to scan result row")
		}
		results = append(results, content)
	}

	if err := rows.Err(); err != nil {
		return nil, errors.Wrapf(err, "error iterating result rows")
	}

	return results, nil
}

// DeleteAgentKnowledge removes all knowledge for an agent
func (idx *sqliteVecIndexer) DeleteAgentKnowledge(ctx context.Context, agentName string) error {
	deleteSQL := `DELETE FROM knowledge_vectors WHERE agent_name = ?`
	if _, err := idx.db.ExecContext(ctx, deleteSQL, agentName); err != nil {
		return errors.Wrapf(err, "failed to delete agent knowledge")
	}
	return nil
}

// Close closes the indexer
func (idx *sqliteVecIndexer) Close() error {
	return idx.db.Close()
}

// processKnowledge converts knowledge maps into indexable text chunks
func (idx *sqliteVecIndexer) processKnowledge(knowledge []map[string]any) []KnowledgeChunk {
	var chunks []KnowledgeChunk

	for _, item := range knowledge {
		// Convert the knowledge item to a searchable text representation
		content := idx.extractTextFromKnowledge(item)
		if content == "" {
			continue
		}

		// Store original metadata as JSON
		metadata, _ := json.Marshal(item)

		chunks = append(chunks, KnowledgeChunk{
			Content:  content,
			Metadata: string(metadata),
		})
	}

	return chunks
}

// extractTextFromKnowledge extracts searchable text from a knowledge map
func (idx *sqliteVecIndexer) extractTextFromKnowledge(item map[string]any) string {
	var textParts []string

	// Common text fields to extract
	textFields := []string{"text", "content", "description", "summary", "title", "name"}

	for _, field := range textFields {
		if value, exists := item[field]; exists {
			if str, ok := value.(string); ok && str != "" {
				textParts = append(textParts, str)
			}
		}
	}

	// If no standard text fields found, try to extract from all string values
	if len(textParts) == 0 {
		for key, value := range item {
			if str, ok := value.(string); ok && str != "" {
				textParts = append(textParts, fmt.Sprintf("%s: %s", key, str))
			}
		}
	}

	return strings.Join(textParts, " ")
}
