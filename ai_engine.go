package agentruntime

import (
	"context"
	"io"
	"log/slog"

	"github.com/jcooky/go-din"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/engine"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/habiliai/agentruntime/internal/tool"
)

type (
	RunRequest      = engine.RunRequest
	RunResponse     = engine.RunResponse
	Conversation    = engine.Conversation
	ToolCall        = engine.ToolCall
	Agent           = entity.Agent
	Participant     = engine.Participant
	MessageExample  = entity.MessageExample
	Tool            = entity.Tool
	ToolConfig      = config.ToolConfig
	GenerateRequest = engine.GenerateRequest

	AIEngine struct {
		engine      engine.Engine
		toolManager tool.Manager

		toolConfig   *ToolConfig
		openAIAPIKey string
		xaiAPIKey    string
		logger       *slog.Logger
		traceVerbose bool
		container    *din.Container
	}
)

func (a *AIEngine) Generate(ctx context.Context, req GenerateRequest, out any, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return a.engine.Generate(ctx, &req, out, opts...)
}

func (a *AIEngine) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	return a.engine.Embed(ctx, texts...)
}

func (a *AIEngine) Run(ctx context.Context, req RunRequest, out any) (*RunResponse, error) {
	return a.engine.Run(ctx, req, out)
}

func (a *AIEngine) CreateAgentFromYaml(ctx context.Context, yamlFile io.Reader) (*Agent, error) {
	yamlCfg, err := config.LoadAgentFromFile(yamlFile)
	if err != nil {
		return nil, err
	}

	for name, mcpServer := range yamlCfg.MCPServers {
		if err := a.toolManager.RegisterMCPTool(ctx, tool.RegisterMCPToolRequest{
			ServerName: name,
			Command:    mcpServer.Command,
			Args:       mcpServer.Args,
			Env:        mcpServer.Env,
		}); err != nil {
			return nil, err
		}
	}
	return a.engine.NewAgentFromConfig(ctx, yamlCfg)
}

func (a *AIEngine) Close() {
	a.container.Close()
}

func NewAIEngine(ctx context.Context, optionFuncs ...func(*AIEngine)) *AIEngine {
	c := din.NewContainer(ctx, din.EnvProd)
	e := &AIEngine{
		container: c,
	}
	for _, f := range optionFuncs {
		f(e)
	}
	if e.openAIAPIKey != "" {
		din.SetT(c, &config.OpenAIConfig{
			APIKey: e.openAIAPIKey,
		})
	}
	if e.xaiAPIKey != "" {
		din.SetT(c, &config.XAIConfig{
			APIKey: e.xaiAPIKey,
		})
	}
	if e.logger != nil {
		din.Set(c, mylog.Key, e.logger)
	}
	if e.toolConfig != nil {
		din.SetT(c, e.toolConfig)
	}
	if e.traceVerbose {
		din.SetT(c, &config.LogConfig{
			TraceVerbose: e.traceVerbose,
		})
	}

	e.engine = din.MustGetT[engine.Engine](c)
	e.toolManager = din.MustGetT[tool.Manager](c)
	return e
}

func WithOpenAIAPIKey(apiKey string) func(e *AIEngine) {
	return func(e *AIEngine) {
		e.openAIAPIKey = apiKey
	}
}

func WithLogger(logger *slog.Logger) func(e *AIEngine) {
	return func(e *AIEngine) {
		e.logger = logger
	}
}

func WithTraceVerbose(traceVerbose bool) func(e *AIEngine) {
	return func(e *AIEngine) {
		e.traceVerbose = traceVerbose
	}
}

func WithXAIAPIKey(apiKey string) func(e *AIEngine) {
	return func(e *AIEngine) {
		e.xaiAPIKey = apiKey
	}
}

func WithToolConfig(toolConfig *config.ToolConfig) func(e *AIEngine) {
	return func(e *AIEngine) {
		e.toolConfig = toolConfig
	}
}

func (a *AIEngine) IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error {
	return a.engine.IndexKnowledge(ctx, agentName, knowledge)
}

func (a *AIEngine) RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error) {
	return a.engine.RetrieveRelevantKnowledge(ctx, agentName, query, limit)
}
