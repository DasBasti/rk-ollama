# Platinenmachergpt - Ollama API for RKLLM

A Rust-based HTTP server that provides an Ollama-compatible API for RKLLM (Rockchip Large Language Model) inference. This project enables you to run RKLLM models through a REST API that mimics Ollama's interface, making it easy to integrate with existing Ollama-compatible clients and applications.

## Features (to be)

- 🚀 **Ollama-Compatible API**: Drop-in replacement for Ollama with the same REST endpoints and request/response formats
- ⚡ **High Performance**: Built with Tokio and Axum for asynchronous, high-throughput request handling
- 🧠 **RKLLM Integration**: Native support for Rockchip's optimized LLM models (.rkllm files)
- 🔄 **Streaming Support**: Real-time response streaming via callback handlers
- 💬 **Chat API**: Full conversational AI with message history support
- ⚙️ **Configurable Parameters**: Support for temperature, top_p, top_k, max_tokens, and other generation parameters
- 🛡️ **Error Handling**: Robust error handling with proper HTTP status codes
- 🔒 **CORS Support**: Cross-origin resource sharing enabled for web applications
- 📝 **JSON API**: Clean JSON request/response format
- 📂 **Dynamic Model Loading**: Automatically load different models based on the `model` field in requests

## Dynamic Model Loading

The server supports loading different `.rkllm` models dynamically based on the `model` name specified in API requests. This allows you to have multiple models available and switch between them without restarting the server.

### How It Works

1. When a request comes in with a `model` field (e.g., `"model": "llama3"`), the server searches for a corresponding `.rkllm` file.
2. The search order is:
   - If `MODEL_PATH` is a directory: look for `<model_name>.rkllm` inside that directory
   - If `MODEL_PATH` is a directory: also check subdirectories named after the model
   - If `MODEL_PATH` is a file: look in its parent directory
   - Current working directory as a fallback
   - `models/` subdirectory in the current working directory

### Example

```bash
# Set MODEL_PATH to a directory containing your models
export MODEL_PATH=/home/user/models

# Directory structure:
# /home/user/models/
# ├── llama3.rkllm
# ├── deepseek-r1.rkllm
# └── gemma3.rkllm

# Now you can request different models:
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3", "prompt": "Hello"}'

curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "deepseek-r1", "prompt": "Hello"}'
```

### Listing Available Models

Use the `/api/tags` endpoint to list all available `.rkllm` models:

```bash
curl http://localhost:11434/api/tags
```

## Prerequisites

- Rust 1.70+ (2024 edition)
- RKLLM-compatible hardware (Rockchip RK3588/RK3588S or similar)
- A `.rkllm` model file

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd platinenmachergpt
   ```

2. **Install dependencies:**
   ```bash
   cargo build --release
   ```

## Logging

The server provides comprehensive logging to help monitor requests and debug issues:

### Log Levels

- **INFO**: General information about server startup and request completion
- **DEBUG**: Detailed information about request processing and parameters
- **WARN**: Non-critical issues (e.g., model destruction failures)
- **ERROR**: Critical errors that prevent request completion

### Configuring Log Levels

Set the log level using the `RUST_LOG` environment variable:

```bash
# Basic logging
RUST_LOG=platinenmachergpt=info

# Detailed logging
RUST_LOG=platinenmachergpt=debug,tower_http=info

# Include all components
RUST_LOG=info
```

### Sample Log Output

```
2025-09-03T10:30:15.123Z INFO  platinenmachergpt: Starting Platinenmachergpt server...
2025-09-03T10:30:15.124Z INFO  platinenmachergpt: Using model path: /root/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm
2025-09-03T10:30:15.125Z INFO  platinenmachergpt: 🚀 Platinenmachergpt server listening on http://0.0.0.0:11434
2025-09-03T10:30:20.456Z INFO  platinenmachergpt::generate_handler: Received generate request for model: rkllm-model
2025-09-03T10:30:20.457Z DEBUG platinenmachergpt::generate_handler: Request details: prompt_length=25, stream=None
2025-09-03T10:30:20.458Z DEBUG platinenmachergpt::generate_handler: Starting RKLLM inference with params: max_tokens=256, top_k=40, top_p=0.9, temperature=0.8
2025-09-03T10:30:22.123Z INFO  platinenmachergpt::generate_handler: RKLLM model initialized successfully
2025-09-03T10:30:25.456Z INFO  platinenmachergpt::generate_handler: RKLLM inference completed successfully
2025-09-03T10:30:25.457Z INFO  platinenmachergpt::generate_handler: Generate request completed successfully, response length: 150
```

The application will automatically load variables from the `.env` file.

### Setting Environment Variables Directly

```bash
export MODEL_PATH="/path/to/your/model.rkllm"
```

## API Endpoints

### Generate Text

**Endpoint:** `POST /api/generate`

Generate text using the loaded RKLLM model.

**Request Body:**
```json
{
  "model": "your-model-name",
  "prompt": "Your prompt text here",
  "stream": false,
  "options": {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 256,
    "repeat_penalty": 1.1
  }
}
```

**Response:**
```json
{
  "model": "your-model-name",
  "response": "Generated text response...",
  "done": true,
  "done_reason": "stop"
}
```

### Chat (Conversational AI)

**Endpoint:** `POST /api/chat`

Chat with the model using conversational format with message history.

**Request Body:**
```json
{
  "model": "your-model-name",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.8,
    "max_tokens": 150
  }
}
```

**Response:**
```json
{
  "model": "your-model-name",
  "message": {
    "role": "assistant",
    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
  },
  "done": true,
  "done_reason": "stop"
}
```

## Usage

### Starting the Server

```bash
# With .env file (recommended)
cargo run

# With custom model path (overrides .env)
MODEL_PATH="/path/to/your/model.rkllm" cargo run
```

The server will start on `http://0.0.0.0:11434` (Ollama's default port) and automatically load configuration from the `.env` file if present.

### Example API Calls

#### Generate Text
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "prompt": "Explain quantum computing in simple terms",
    "options": {
      "temperature": 0.7,
      "max_tokens": 150
    }
  }'
```

#### Chat
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms"
      }
    ],
    "options": {
      "temperature": 0.7,
      "max_tokens": 150
    }
  }'
```

### Integration with Ollama Clients

Since this server provides an Ollama-compatible API, you can use any Ollama client:

```bash
# Using ollama CLI (if configured to point to your server)
ollama run your-model "Hello world"

# Using Python requests
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "your-model",
    "prompt": "Hello world"
})
print(response.json())
```

## Architecture

### Core Components

- **Server**: Axum-based HTTP server with async request handling
- **Model Integration**: RKLLM Rust wrapper for model loading and inference
- **Callback Handler**: Custom handler for collecting streaming responses
- **Parameter Management**: Configurable generation parameters with defaults

### Async Architecture

The server uses Tokio for async runtime and `spawn_blocking` for RKLLM operations to prevent blocking the main thread:

```
HTTP Request → Axum Handler → spawn_blocking → RKLLM Inference → Callback → Response
```

### Key Files

- `src/main.rs`: Main server implementation with API endpoints
- `Cargo.toml`: Dependencies and project configuration
- `README.md`: This documentation

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

### Testing

```bash
# Run tests
cargo test

# Check code
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

### Dependencies

- `tokio`: Async runtime
- `axum`: Web framework
- `serde`: JSON serialization
- `tower-http`: HTTP middleware (CORS)
- `rkllm-rs`: RKLLM Rust wrapper
- `chrono`: Date/time handling

## Troubleshooting

### Model Loading Issues

- Ensure `MODEL_PATH` points to a valid `.rkllm` file
- Check file permissions
- Verify RKLLM-compatible hardware

### Server Won't Start

- Check if port 11434 is available
- Verify all dependencies are installed
- Check for compilation errors

### API Errors

- Ensure request JSON is properly formatted
- Check model path configuration
- Verify RKLLM model compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Rockchip for RKLLM
- Ollama for the API specification
- The Rust community for excellent async libraries
