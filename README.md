# Simple MCP Client

A multi-modal AI chat client with MCP (Model Context Protocol) tool integration.

hosted at: https://mcp-client-chat.valiantlynx.com

The chat agent at [agents/chat_agent.py](https://github.com/mythrantic/mcp-client-chat/blob/main/src/mcp_client_chat/agents/chat_agent.py) is made using [machine-core](https://github.com/samletnorge/machine-core). An AI agents orchistration abstraction

## Architecture

This project follows clean architecture principles:

- **Core**: `an agent from machine-core` - business logic and MCP client
- **Frontend**: `app.py` - Streamlit web interface

## Features

- 🤖 Multi-modal AI chat (text + images)
- 🔧 MCP tool integration via `mcp.json`
- 📝 Markdown rendering with support for images and rich content
- 🖼️ Multiple image upload support
- 📊 Token usage tracking
- 🔄 Streaming responses
- ⚙️ Configurable model and MCP servers

## Installation

```bash
# Install dependencies
uv sync
```

## Usage

### Streamlit Web Interface

```bash
# Run the web app
uv run streamlit run src/mcp_client_chat/app.py

# Or with custom port
uv run streamlit run src/mcp_client_chat/app.py --server.port 8501
```

Then open http://localhost:8501 in your browser.

### Command Line Interface

```bash
# Single query
uv run python src/mcp_client_chat/client.py "What is the weather?"

# With single image
uv run python src/mcp_client_chat/client.py "Describe this image" -i image.png

# With multiple images
uv run python src/mcp_client_chat/client.py "Compare these images" -i img1.png -i img2.jpg

# With image URL
uv run python src/mcp_client_chat/client.py "What's in this?" -i https://example.com/image.png

# With data URL
uv run python src/mcp_client_chat/client.py "Analyze" -i "data:image/jpeg;base64,/9j/4AAQ..."
```

## Configuration

### MCP Servers (`mcp.json`)

Configure MCP servers in VS Code format:

```json
{
  "servers": {
    "playwright": {
      "type": "http",
      "url": "https://playwright-mcp.valiantlynx.com/mcp"
    },
    "task-manager": {
      "type": "http",
      "url": "http://127.0.0.1:8081/mcp"
    },
    "notifier": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "path/to/notify.py"]
    }
  }
}
```

### Environment Variables

Set your LLM provider configuration in `.env` or environment:

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Or OpenAI
OPENAI_API_KEY=your-key
```

## Project Structure

```
mcp-client-chat/
├── src/mcp_client_chat/
│   ├── agents/       # Core business logic. a simple chat agent
│   ├── app.py          # Streamlit frontend
│   ├── config.py       # Configuration & prompts
│   └── __init__.py
├── mcp.json            # MCP server configuration
├── pyproject.toml      # Dependencies
└── README.md
```

## Image Support

The client supports three types of image inputs:

1. **Local files**: `./image.png`
2. **HTTP/HTTPS URLs**: `https://example.com/image.png`
3. **Data URLs**: `data:image/jpeg;base64,...`

All image types are automatically processed and converted to base64 data URLs for the model.

## Development

### Core Client (`client.py`)

The `MCPClient` class provides:

- `__init__(model_name, tools_urls, mcp_config_path)` - Initialize client
- `process_image(image_path)` - Process and encode images
- `run_query(query, image_paths)` - Execute queries with optional images

### Frontend (`app.py`)

The Streamlit app provides:

- Chat interface with markdown rendering
- Image upload and preview
- MCP client configuration
- Token usage tracking
- Chat history management

## License

MIT
