<p align="center">
  <img alt="Shows a white agents.json Logo with a black background." src="https://u6mo491ntx4iwuoz.public.blob.vercel-storage.com/logo/bg_black_logo-tzo7s5eNJEWkXMEVBMME7ucb7BUN2L.png" width="full">
</p>

<h1 align="center">AI Agent Runtime by <i>Habili.ai</i> </h1>

[![Go Build & Test Pipeline](https://github.com/habiliai/agentruntime/actions/workflows/ci.yml/badge.svg)](https://github.com/habiliai/agentruntime/actions/workflows/ci.yml)
[![Go Lint Pipeline](https://github.com/habiliai/agentruntime/actions/workflows/lint.yml/badge.svg)](https://github.com/habiliai/agentruntime/actions/workflows/lint.yml)

## Overview

`agentruntime` is a comprehensive platform for deploying AI agents in a local environment. It provides a unified runtime for various LLM-powered agents with different capabilities and tools.

### Key Features

- **Genkit Integration**: Seamlessly integrate with the Genkit platform for agent development and deployment
- **Single Binary Executable**: Run agents with a portable, self-contained binary that requires no external dependencies
- **Simple Agent Configuration**: Define agent capabilities, tools, and behavior through intuitive YAML configuration
- **Tool Extensibility**: Easily extend agent capabilities with custom tools and integrations
- **Thread Management**: Maintain conversation state and history across multiple interactions
- **Agent Orchestration**: Coordinate multiple agents working together to solve complex tasks
- **RAG (Retrieval-Augmented Generation)**: Dynamic knowledge retrieval using vector embeddings for context-aware responses

### RAG (Retrieval-Augmented Generation) Support

AgentRuntime includes built-in RAG functionality powered by [sqlite-vec](https://github.com/asg017/sqlite-vec), enabling agents to have dynamic, context-aware access to their knowledge base.

#### Features
- **Automatic Knowledge Indexing**: Agent knowledge is automatically indexed into vector embeddings when agents are created
- **Context-Aware Retrieval**: Relevant knowledge is retrieved based on conversation context
- **Vector Similarity Search**: Uses OpenAI embeddings with sqlite-vec for fast, accurate retrieval
- **Graceful Degradation**: Falls back to static knowledge if RAG is unavailable

#### Quick Setup

1. **Install sqlite-vec extension**:
   ```bash
   # Use the provided installation script
   ./scripts/install-sqlite-vec.sh
   
   # Or install manually from https://github.com/asg017/sqlite-vec/releases
   ```

2. **Configure agent with knowledge**:
   ```yaml
   name: TravelAgent
   model: openai/gpt-4o
   knowledge:
     - cityName: Seoul
       aliases: Seoul, SEOUL, KOR, Korea
       info: Capital city of South Korea
     - cityName: Tokyo
       aliases: Tokyo, TYO, Japan
       info: Capital city of Japan
   ```

3. **Knowledge is automatically indexed and retrieved during conversations**:
   - When the agent is created, knowledge is indexed into vector embeddings
   - During conversations, relevant knowledge is retrieved based on context
   - Retrieved knowledge is injected into the agent's prompt for accurate responses

For detailed RAG documentation, see [internal/rag/README.md](internal/rag/README.md).

The platform consists of three core components:

1. **AgentRuntime**: The main execution environment that orchestrates all agent activities
2. **AgentNetwork**: The network server that manages agent registration, health checks, and communication between agents and the runtime

## Installation

### Option 1: Download pre-built binaries (Recommended)

```bash
# For macOS
# Install agentnetwork
curl -L https://github.com/habiliai/agentruntime/releases/latest/download/agentnetwork-darwin-amd64 -o agentnetwork
chmod +x agentnetwork
sudo mv agentnetwork /usr/local/bin/

# Install agentruntime
curl -L https://github.com/habiliai/agentruntime/releases/latest/download/agentruntime-darwin-amd64 -o agentruntime
chmod +x agentruntime
sudo mv agentruntime /usr/local/bin/

# For Linux
# Install agentnetwork
curl -L https://github.com/habiliai/agentruntime/releases/latest/download/agentnetwork-linux-amd64 -o agentnetwork
chmod +x agentnetwork
sudo mv agentnetwork /usr/local/bin/

# Install agentruntime
curl -L https://github.com/habiliai/agentruntime/releases/latest/download/agentruntime-linux-amd64 -o agentruntime
chmod +x agentruntime
sudo mv agentruntime /usr/local/bin/

# For Windows (using PowerShell)
# Install agentnetwork
Invoke-WebRequest -Uri https://github.com/habiliai/agentruntime/releases/latest/download/agentnetwork-windows-amd64.exe -OutFile agentnetwork.exe

# Install agentruntime
Invoke-WebRequest -Uri https://github.com/habiliai/agentruntime/releases/latest/download/agentruntime-windows-amd64.exe -OutFile agentruntime.exe
```

You can also download the binaries directly from the [releases page](https://github.com/habiliai/agentruntime/releases).

### Option 2: Build from source

Prerequisites:
- Go 1.21 or higher
- Make

```bash
# Clone the repository
git clone https://github.com/habiliai/agentruntime.git
cd agentruntime

# Build the project
make build
```

## Quick Start

1. Create a basic agent configuration file (example.agent.yaml):

```yaml
name: BasicAgent
version: v1
description: A simple demo agent
tools:
  - name: search
    description: Search the web for information
  - name: calculator
    description: Perform mathematical calculations
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
```

2. Create a `.env` file from the provided example:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# Replace YOUR_API_KEY with your actual keys
nano .env  # or use any text editor
```

Example `.env` file content:
```
LOG_LEVEL=debug
HOST=0.0.0.0
PORT=8010
DATABASE_URL=postgres://postgres:postgres@postgres.local:5432/postgres?sslmode=disable&search_path=agentruntime
# OpenAI API Config
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
# OpenWeather API key
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY
```

3. Start the network server:

```bash
# Start the agentnetwork server
bin/agentnetwork serve
```

4. In a separate terminal, create a thread and run the agent:

```bash
# Create a new thread
bin/agentnetwork thread create
# Run your agent
bin/agentruntime <agent files or directory>
```

5. Interact with your agent through the command-line interface:

```bash
# Connect to the thread to see ongoing interactions
bin/agentnetwork connect <thread id>
```

### Runtime Service

The Runtime service is responsible for executing agents in the context of a thread.

```json
// JSON-RPC Method: habiliai-agentruntime-v1.Run
{
  "jsonrpc": "2.0",
  "method": "habiliai-agentruntime-v1.Run",
  "params": {
    "thread_id": 123,
    "agent_names": ["agent1", "agent2"]
  },
  "id": 1
}
```

#### CLI Usage

```bash
# Run one or more agents in a thread
agentruntime run <thread_id> <agent_name> [<agent_name2> <agent_name3> ...]
```

### ThreadManager Service

The ThreadManager service handles conversation threads and messages through JSON-RPC methods.

```json
// JSON-RPC Methods:
// habiliai-agentnetwork-v1.CreateThread
// habiliai-agentnetwork-v1.GetThread
// habiliai-agentnetwork-v1.AddMessage
// habiliai-agentnetwork-v1.GetMessages
// habiliai-agentnetwork-v1.GetNumMessages

// Example: Create Thread
{
  "jsonrpc": "2.0",
  "method": "habiliai-agentnetwork-v1.CreateThread",
  "params": {
    "instruction": "You are a helpful assistant"
  },
  "id": 1
}

// Example: Add Message
{
  "jsonrpc": "2.0",
  "method": "habiliai-agentnetwork-v1.AddMessage",
  "params": {
    "thread_id": 123,
    "sender": "user",
    "content": "Hello, how are you?",
    "tool_calls": []
  },
  "id": 2
}
```

#### CLI Usage

```bash
# Create a new thread
agentruntime thread create [--instruction "Your instruction"]

# Add a message to a thread
agentruntime thread add-message <thread_id> "Your message"

# List all threads
agentruntime thread list

# List messages in a thread
agentruntime thread list-messages <thread_id>
```

### AgentNetwork Service

The AgentNetwork service is the main network server that coordinates all services including ThreadManager, AgentManager, and Runtime through JSON-RPC.

```json
// JSON-RPC Methods:
// habiliai-agentnetwork-v1.CheckLive
// habiliai-agentnetwork-v1.RegisterAgent
// habiliai-agentnetwork-v1.DeregisterAgent
// habiliai-agentnetwork-v1.GetAgentRuntimeInfo

// Example: Register Agent
{
  "jsonrpc": "2.0",
  "method": "habiliai-agentnetwork-v1.RegisterAgent",
  "params": {
    "addr": "localhost:8080",
    "info": [{
      "name": "myagent",
      "role": "assistant",
      "metadata": {}
    }]
  },
  "id": 1
}

// Example: Get Agent Runtime Info
{
  "jsonrpc": "2.0",
  "method": "habiliai-agentnetwork-v1.GetAgentRuntimeInfo",
  "params": {
    "names": ["myagent"],
    "all": false
  },
  "id": 2
}
```

#### CLI Usage

```bash
# Start the network server
agentnetwork serve [--config <config-path>]

# Connect to a network server
agentnetwork connect [--addr <addr>]

# List threads in the network
agentnetwork thread list

# Create a thread on the network
agentnetwork thread create

# Add a message to a thread
agentnetwork thread add-message <thread_id> "Your message"
```

### Server Mode

AgentRuntime has been split into two main commands:

1. **agentnetwork** - Handles the network server that manages threads and agents
2. **agentruntime** - Handles agent execution and interaction

#### Starting the Network Server

```bash
# Start the network server
agentnetwork serve

# With file watching (auto-reload when config changes)
agentnetwork serve --watch <agent-file-or-directory>
```

#### Connecting to a Network Server

```bash
# Connect to the default network server
agentnetwork connect

# Connect to a specific network server
agentnetwork connect --host <host> --port <port>
```

#### Running Agents

```bash
# Run multiple agents
agentruntime <agent files or directory> [...<agent files or directory>]
```

When running in server mode, clients can connect to the JSON-RPC endpoints to use the services programmatically. The server listens on HTTP and accepts standard JSON-RPC 2.0 requests.

## License

This project is licensed under the MIT License. see the [LICENSE](LICENSE) file for details.
