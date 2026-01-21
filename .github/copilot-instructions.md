# Platinenmachergpt - AI Coding Assistant Instructions

## Project Overview
Rust-based HTTP server providing Ollama-compatible API for RKLLM (Rockchip Large Language Model) inference. Key insight: this bridges RKLLM hardware acceleration with standard Ollama API compatibility, enabling VS Code Copilot and other clients.

## Architecture Patterns

### Async + Blocking Hybrid Pattern
**Critical**: RKLLM operations are CPU-intensive and block threads. Always use `tokio::task::spawn_blocking`:
```rust
let result = tokio::task::spawn_blocking(move || {
    // RKLLM operations here (rkllm_init, llm_handle.run, etc.)
    let llm_handle = rkllm_init(&mut param)?;
    // ... inference work
    llm_handle.destroy()?;
}).await;
```

### Model Lifecycle Pattern (Per-Request)
- **Current**: Models initialized/destroyed per request (see `generate_handler`, `chat_handler`)
- **Why**: Avoids memory leaks, simpler state management
- **Trade-off**: Higher latency for initialization overhead
- Models loaded from `MODEL_PATH` env var each time

### Callback-Based Streaming
RKLLM uses callback handlers for response streaming:
```rust
struct ApiCallbackHandler {
    response: Arc<Mutex<String>>,  // Collects streamed tokens
}
impl RkllmCallbackHandler for ApiCallbackHandler {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        // Only collect on LLMCallState::Normal
    }
}
```

### Dual API Compatibility 
Supports both:
- Ollama format: `/api/generate`, `/api/chat` (native)
- OpenAI format: `/v1/chat/completions` (converts to Ollama internally via `openai_chat_handler`)

## Development Workflow

### Environment Setup
1. Copy `.env.example` to `.env`, set `MODEL_PATH` to `.rkllm` file
2. Requires RKLLM-compatible hardware (RK3588/RK3588S)
3. Build: `cargo build` (debug) / `cargo build --release`
4. Run: `RUST_LOG=debug cargo run &` (comprehensive logging and run in background)

### Testing Strategy
- **Manual**: Use `./test_ollama_api.sh` for full API validation
- **Quick**: `curl_commands.md` has ready-to-use curl examples  
- **Benchmarking**: `./benchmark_ollama.sh` for performance testing
- **VS Code Integration**: Mock responses enabled for short prompts (see `generate_handler` mock logic)

### Logging Conventions
Uses structured logging with request lifecycle markers:
```rust
info!("=== GENERATE REQUEST START ===");
debug!("Full prompt preview: {}", truncated_prompt);
info!("=== GENERATE REQUEST END ===");
```
Enable with: `RUST_LOG=platinenmachergpt=debug,tower_http=info`

## Code Patterns

### Request Handler Structure
All handlers follow this pattern:
1. Log request details with `#[instrument(skip(state))]`
2. Extract/validate parameters with defaults (e.g., `temperature: 0.8`)
3. Create `ApiCallbackHandler` for response collection
4. Spawn blocking task for RKLLM operations
5. Handle errors with structured logging

### Error Handling
- RKLLM failures return `StatusCode::INTERNAL_SERVER_ERROR`
- Model not found returns `404`
- Always log context: model name, prompt length, etc.
- Use structured error messages with request context

### Configuration Management
- Environment variables via `dotenv()` loading `.env` 
- `MODEL_PATH`: Path to `.rkllm` model file (critical)
- Hardcoded defaults for inference params (see `GenerateOptions`)

### Mock Response Logic
Short test prompts (`<= 10/20 chars` with "hi"/"test"/"hello") return mock responses for development/testing compatibility.

## Integration Points

### RKLLM Rust Wrapper (`rkllm-rs`)
- External crate dependency, likely local: `path="../rkllm-rs"`
- Core types: `RKLLMParam`, `RKLLMInput`, `RKLLMInferParam`, `LLMHandle`
- Must manage lifecycle: init → run → destroy pattern

### External APIs
- Ollama compatibility: mirrors response structures exactly
- OpenAI compatibility: transforms requests/responses between formats
- CORS enabled for web clients

### Port & Service
- Default: `0.0.0.0:11434` (Ollama standard)
- Single binary deployment, no microservices

## Common Gotchas

1. **Memory Management**: Always call `llm_handle.destroy()` to prevent leaks
2. **Blocking Operations**: Never call RKLLM functions in async context directly
3. **Model Path**: Server won't start if `MODEL_PATH` is invalid - check early
4. **Token Estimation**: Uses rough `chars/4` approximation for prompt/completion tokens
5. **Chat Format**: Converts chat messages to single prompt with role prefixes

## File Organization
- `src/main.rs`: Single-file application (all handlers, structs, main)
- Testing: `test_ollama_api.sh`, `benchmark_ollama.sh`, `curl_commands.md`
- Config: `.env` for model path, `Cargo.toml` for dependencies
- Documentation: `README.md` (comprehensive), `README_TESTS.md` (testing guide)
