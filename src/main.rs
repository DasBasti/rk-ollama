use axum::{
    routing::{post, get},
    Router,
    Json,
    extract::State,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use rkllm_rs::prelude::*;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;
use dotenv::dotenv;
use tracing::{info, warn, error, debug};
use regex::Regex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

/// Resolve the model path based on the model name from the request.
/// Searches in MODEL_PATH directory (if it's a directory) or the current working directory
/// for a file named `<model_name>.rkllm`.
/// 
/// Returns the full path to the model file if found, or an error message.
fn resolve_model_path(model_name: &str, base_model_path: &str) -> Result<PathBuf, String> {
    // Clean the model name (remove any path separators for security)
    let clean_model_name = model_name
        .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "")
        .trim()
        .to_string();
    
    if clean_model_name.is_empty() {
        return Err("Model name cannot be empty".to_string());
    }
    
    // Build the expected filename
    let model_filename = if clean_model_name.ends_with(".rkllm") {
        clean_model_name.clone()
    } else {
        format!("{}.rkllm", clean_model_name)
    };
    
    debug!("Looking for model file: {}", model_filename);
    
    // Check if MODEL_PATH is a directory or a file
    let base_path = Path::new(base_model_path);
    
    // Search locations in order of priority:
    // 1. If MODEL_PATH is a directory, look for the model file there
    // 2. Current working directory
    // 3. MODEL_PATH parent directory (if MODEL_PATH is a file)
    
    let mut search_paths: Vec<PathBuf> = Vec::new();
    
    if base_path.is_dir() {
        // MODEL_PATH is a directory - look for model file inside it
        search_paths.push(base_path.join(&model_filename));
        // Also check subdirectories with the model name
        search_paths.push(base_path.join(&clean_model_name).join(&model_filename));
    } else if base_path.is_file() {
        // MODEL_PATH is a file - check its parent directory
        if let Some(parent) = base_path.parent() {
            search_paths.push(parent.join(&model_filename));
        }
    }
    
    // Add current working directory as fallback
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.join(&model_filename));
        // Also check a 'models' subdirectory in cwd
        search_paths.push(cwd.join("models").join(&model_filename));
    }
    
    // Try each search path
    for path in &search_paths {
        debug!("Checking path: {:?}", path);
        if path.exists() && path.is_file() {
            info!("Found model file at: {:?}", path);
            return Ok(path.clone());
        }
    }
    
    // If not found, return a helpful error message
    Err(format!(
        "Model '{}' not found. Searched for '{}' in: {:?}",
        model_name,
        model_filename,
        search_paths
    ))
}

/// List all available .rkllm model files in the search paths
fn list_available_models(base_model_path: &str) -> Vec<String> {
    let mut models = Vec::new();
    let mut search_dirs: Vec<PathBuf> = Vec::new();
    
    let base_path = Path::new(base_model_path);
    
    if base_path.is_dir() {
        search_dirs.push(base_path.to_path_buf());
    } else if let Some(parent) = base_path.parent() {
        if parent.is_dir() {
            search_dirs.push(parent.to_path_buf());
        }
    }
    
    if let Ok(cwd) = std::env::current_dir() {
        search_dirs.push(cwd.clone());
        let models_dir = cwd.join("models");
        if models_dir.is_dir() {
            search_dirs.push(models_dir);
        }
    }
    
    for dir in search_dirs {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "rkllm" {
                            if let Some(stem) = path.file_stem() {
                                let model_name = stem.to_string_lossy().to_string();
                                if !models.contains(&model_name) {
                                    models.push(model_name);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    models
}

/// Clean response content by removing thinking tags and extra whitespace
fn clean_response_content(content: &str) -> String {
    // Remove complete <think>...</think> blocks (case insensitive, multiline)
    let think_complete_regex = Regex::new(r"(?si)<think\s*>.*?</think\s*>").unwrap();
    let cleaned = think_complete_regex.replace_all(content, "");
    
    // Remove incomplete <think> blocks (everything from <think> to end if no closing tag)
    let think_incomplete_regex = Regex::new(r"(?si)<think\s*>.*$").unwrap();
    let cleaned = think_incomplete_regex.replace_all(&cleaned, "");
    
    // Clean up extra whitespace and newlines
    let whitespace_regex = Regex::new(r"\n\s*\n\s*\n").unwrap();
    let trimmed = whitespace_regex.replace_all(&cleaned, "\n\n");
    
    // Final cleanup: trim and ensure no leading/trailing whitespace
    let result = trimmed.trim().to_string();
    
    // If result is empty or very short, provide a fallback response
    if result.is_empty() || result.len() < 10 {
        "I understand your request. How can I help you?".to_string()
    } else {
        result
    }
}

#[derive(Deserialize, Debug)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: Option<bool>,
    options: Option<GenerateOptions>,
}

#[derive(Deserialize, Debug)]
struct GenerateOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    max_tokens: Option<i32>,
    repeat_penalty: Option<f32>,
}

#[derive(Serialize)]
struct GenerateResponse {
    model: String,
    response: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: Option<bool>,
    options: Option<GenerateOptions>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    model: String,
    created_at: String,
    message: Message,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u64>,
}

#[derive(Serialize)]
struct VersionResponse {
    version: String,
}

#[derive(Serialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct RunningModelsResponse {
    models: Vec<RunningModelInfo>,
}

#[derive(Serialize)]
struct RunningModelInfo {
    name: String,
    model: String,
    size: u64,
    digest: String,
    details: ModelDetails,
    expires_at: String,
    size_vram: u64,
}

#[derive(Deserialize, Debug)]
struct ShowRequest {
    model: String,
    #[serde(default)]
    verbose: Option<bool>,
}

// OpenAI-compatible API structures
#[derive(Deserialize, Debug)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OpenAIChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Serialize)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ShowResponse {
    modelfile: String,
    parameters: String,
    template: String,
    details: ModelDetails,
    model_info: ModelInfo,
    capabilities: Vec<String>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: ModelDetails,
}

#[derive(Serialize)]
struct ModelDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Option<Vec<String>>,
    parameter_size: String,
    quantization_level: String,
}

// RKLLM callback handler for collecting responses
struct ApiCallbackHandler {
    response: Arc<Mutex<String>>,
}

impl RkllmCallbackHandler for ApiCallbackHandler {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        match state {
            LLMCallState::Normal => {
                if let Some(result) = result {
                    let mut response = self.response.blocking_lock();
                    response.push_str(&result.text);
                }
            }
            LLMCallState::Waiting => {}
            LLMCallState::Finish => {}
            LLMCallState::Error => {
                let mut response = self.response.blocking_lock();
                response.push_str("Error occurred during generation");
            }
            LLMCallState::GetLastHiddenLayer => {}
        }
    }
}

// Application state holding the LLM handle
#[derive(Clone)]
struct AppState {
    llm_handle: Arc<Mutex<Option<LLMHandle>>>,
    model_path: String,
    max_context_len: i32,
    max_new_tokens_default: i32,
    max_new_tokens_limit: i32,
    max_prompt_length: usize,
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    info!("=== GENERATE REQUEST START ===");
    info!("Received generate request for model: '{}'", req.model);
    info!("Request details: prompt_length={}, stream={:?}", req.prompt.len(), req.stream);
    debug!("Full prompt preview: {}", if req.prompt.len() > 100 { 
        format!("{}...", &req.prompt[..100]) 
    } else { 
        req.prompt.clone() 
    });

    // Resolve the model path based on the model name in the request
    let resolved_model_path = match resolve_model_path(&req.model, &state.model_path) {
        Ok(path) => {
            info!("Resolved model '{}' to path: {:?}", req.model, path);
            path.to_string_lossy().to_string()
        }
        Err(e) => {
            error!("Failed to resolve model '{}': {}", req.model, e);
            return Err(StatusCode::NOT_FOUND);
        }
    };

    // Validate prompt length
    if req.prompt.len() > state.max_prompt_length {
        error!("Prompt too long: {} characters > {} limit", req.prompt.len(), state.max_prompt_length);
        return Err(StatusCode::BAD_REQUEST);
    }

    let response_text = Arc::new(Mutex::new(String::new()));
    let callback = ApiCallbackHandler {
        response: Arc::clone(&response_text),
    };

    // Run inference in a blocking task
    let model_path = resolved_model_path;
    let prompt = req.prompt.clone();
    let requested_max_tokens = req.options.as_ref().and_then(|o| o.max_tokens).unwrap_or(state.max_new_tokens_default);
    let max_tokens = std::cmp::min(requested_max_tokens, state.max_new_tokens_limit);
    
    // Estimate prompt tokens (rough approximation: ~4 chars per token)
    let estimated_prompt_tokens = (req.prompt.len() / 4) as i32;
    
    // Ensure prompt + max_tokens doesn't exceed context length
    let max_tokens = if estimated_prompt_tokens + max_tokens > state.max_context_len {
        let available_tokens = state.max_context_len - estimated_prompt_tokens;
        if available_tokens <= 0 {
            error!("Prompt too long for context: estimated {} tokens > {} context limit", 
                   estimated_prompt_tokens, state.max_context_len);
            return Err(StatusCode::BAD_REQUEST);
        }
        let adjusted_max_tokens = std::cmp::min(max_tokens, available_tokens);
        warn!("Adjusted max_tokens from {} to {} to fit context window (estimated prompt: {} tokens)", 
              max_tokens, adjusted_max_tokens, estimated_prompt_tokens);
        adjusted_max_tokens
    } else {
        max_tokens
    };
    
    let top_k = req.options.as_ref().and_then(|o| o.top_k).unwrap_or(40);
    let top_p = req.options.as_ref().and_then(|o| o.top_p).unwrap_or(0.9);
    let temperature = req.options.as_ref().and_then(|o| o.temperature).unwrap_or(0.8);
    let repeat_penalty = req.options.as_ref().and_then(|o| o.repeat_penalty).unwrap_or(1.1);
    let max_context_len = state.max_context_len;

    info!("Generate parameters: max_tokens={}, top_k={}, top_p={:.2}, temperature={:.2}, repeat_penalty={:.2}",
           max_tokens, top_k, top_p, temperature, repeat_penalty);
    debug!("Starting RKLLM inference with params: max_tokens={}, top_k={}, top_p={}, temperature={}",
           max_tokens, top_k, top_p, temperature);

    let result = tokio::task::spawn_blocking(move || {
        let start_time = std::time::Instant::now();
        debug!("Initializing RKLLM model...");
        // Create parameters inside the blocking task
        let model_path_cstr = std::ffi::CString::new(model_path)?;
        let mut param = RKLLMParam {
            model_path: model_path_cstr.as_ptr() as *const std::os::raw::c_char,
            max_context_len: max_context_len,
            max_new_tokens: max_tokens,
            top_k: top_k,
            top_p: top_p,
            temperature: temperature,
            repeat_penalty: repeat_penalty,
            ..Default::default()
        };

        let infer_params = RKLLMInferParam {
            mode: RKLLMInferMode::InferGenerate,
            lora_params: None,
            prompt_cache_params: None,
            keep_history: KeepHistory::NoKeepHistory,
        };

        let load_start = std::time::Instant::now();
        let llm_handle = match rkllm_init(&mut param) {
            Ok(handle) => {
                info!("RKLLM model initialized successfully");
                handle
            }
            Err(e) => {
                error!("Failed to initialize RKLLM model: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("RKLLM init failed: {:?}", e)));
            }
        };
        let load_duration = load_start.elapsed().as_nanos() as u64;

        let input = RKLLMInput {
            input_type: RKLLMInputType::Prompt(prompt.clone()),
            enable_thinking: false,
            role: RKLLMInputRole::User,
        };

        let prompt_eval_start = std::time::Instant::now();
        debug!("Running RKLLM inference...");
        let run_result = llm_handle.run(
            input,
            Some(infer_params),
            callback,
        );
        let prompt_eval_duration = prompt_eval_start.elapsed().as_nanos() as u64;

        match run_result {
            Ok(_) => info!("RKLLM inference completed successfully"),
            Err(e) => {
                error!("RKLLM inference failed: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("RKLLM run failed: {:?}", e)));
            }
        }

        match llm_handle.destroy() {
            Ok(_) => debug!("RKLLM handle destroyed successfully"),
            Err(e) => warn!("Failed to destroy RKLLM handle: {:?}", e),
        }

        let total_duration = start_time.elapsed().as_nanos() as u64;
        
        // Estimate token counts (rough approximation)
        let prompt_eval_count = (prompt.len() / 4) as u32; // Rough estimate: ~4 chars per token
        let response_len = response_text.blocking_lock().len();
        let eval_count = (response_len / 4) as u32; // Rough estimate
        let eval_duration = total_duration - load_duration - prompt_eval_duration;

        Ok((response_text.blocking_lock().clone(), load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration))
    }).await;

    match result {
        Ok(Ok((response, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration))) => {
            let response_preview = if response.len() > 100 { 
                format!("{}...", &response[..100]) 
            } else { 
                response.clone() 
            };
            
            info!("Generate request completed successfully!");
            info!("Response stats: length={} chars, estimated_tokens={}", response.len(), eval_count);
            info!("Timing: total={:.2}ms, load={:.2}ms, prompt_eval={:.2}ms, generation={:.2}ms", 
                  total_duration as f64 / 1_000_000.0,
                  load_duration as f64 / 1_000_000.0,
                  prompt_eval_duration as f64 / 1_000_000.0,
                  eval_duration as f64 / 1_000_000.0);
            debug!("Generated response preview: {}", response_preview);
            info!("=== GENERATE REQUEST END ===");
            
            Ok(Json(GenerateResponse {
                model: req.model,
                response,
                done: true,
                done_reason: Some("stop".to_string()),
                total_duration: Some(total_duration),
                load_duration: Some(load_duration),
                prompt_eval_count: Some(prompt_eval_count),
                prompt_eval_duration: Some(prompt_eval_duration),
                eval_count: Some(eval_count),
                eval_duration: Some(eval_duration),
            }))
        }
        Ok(Err(e)) => {
            error!("=== GENERATE REQUEST FAILED ===");
            error!("Inference task failed: {}", e);
            error!("Model: {}, Prompt length: {}", req.model, req.prompt.len());
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(e) => {
            error!("=== GENERATE REQUEST FAILED ===");
            error!("Task spawn failed: {}", e);
            error!("Model: {}, Prompt length: {}", req.model, req.prompt.len());
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    info!("=== CHAT REQUEST START ===");
    info!("Received chat request for model: '{}'", req.model);
    info!("Chat details: {} messages, stream={:?}", req.messages.len(), req.stream);
    
    // Resolve the model path based on the model name in the request
    let resolved_model_path = match resolve_model_path(&req.model, &state.model_path) {
        Ok(path) => {
            info!("Resolved model '{}' to path: {:?}", req.model, path);
            path.to_string_lossy().to_string()
        }
        Err(e) => {
            error!("Failed to resolve model '{}': {}", req.model, e);
            return Err(StatusCode::NOT_FOUND);
        }
    };
    
    // Log message summary
    for (i, message) in req.messages.iter().enumerate() {
        let content_preview = if message.content.len() > 50 { 
            format!("{}...", &message.content[..50]) 
        } else { 
            message.content.clone() 
        };
        debug!("Message {}: role='{}', content_length={}, preview='{}'", 
               i + 1, message.role, message.content.len(), content_preview);
    }
    
    debug!("Chat request with {} messages, stream={:?}", req.messages.len(), req.stream);

    // Convert chat messages to a natural conversation format for the model
    let mut conversation_parts = Vec::new();

    for message in &req.messages {
        match message.role.as_str() {
            "system" => {
                // System messages can be included naturally
                conversation_parts.push(message.content.clone());
            }
            "user" | "assistant" => {
                // Include user and assistant messages as natural conversation
                conversation_parts.push(message.content.clone());
            }
            _ => {
                debug!("Unknown message role: {}, including content as-is", message.role);
                conversation_parts.push(message.content.clone());
            }
        }
    }

    // Create a natural conversation prompt without artificial role prefixes
    let prompt = conversation_parts.join("\n\n");
    info!("Converted {} messages to natural conversation prompt (length: {} chars)", req.messages.len(), prompt.len());
    debug!("Conversation prompt preview: {}", if prompt.len() > 200 { 
        format!("{}...", &prompt[..200]) 
    } else { 
        prompt.clone() 
    });

    // Validate prompt length
    if prompt.len() > state.max_prompt_length {
        error!("Chat prompt too long: {} characters > {} limit", prompt.len(), state.max_prompt_length);
        return Err(StatusCode::BAD_REQUEST);
    }

    let response_text = Arc::new(Mutex::new(String::new()));
    let callback = ApiCallbackHandler {
        response: Arc::clone(&response_text),
    };

    // Run inference in a blocking task
    let model_path = resolved_model_path;
    let requested_max_tokens = req.options.as_ref().and_then(|o| o.max_tokens).unwrap_or(state.max_new_tokens_default);
    let max_tokens = std::cmp::min(requested_max_tokens, state.max_new_tokens_limit);
    
    // Estimate prompt tokens (rough approximation: ~4 chars per token)
    let estimated_prompt_tokens = (prompt.len() / 4) as i32;
    
    // Ensure prompt + max_tokens doesn't exceed context length
    let max_tokens = if estimated_prompt_tokens + max_tokens > state.max_context_len {
        let available_tokens = state.max_context_len - estimated_prompt_tokens;
        if available_tokens <= 0 {
            error!("Chat prompt too long for context: estimated {} tokens > {} context limit", 
                   estimated_prompt_tokens, state.max_context_len);
            return Err(StatusCode::BAD_REQUEST);
        }
        let adjusted_max_tokens = std::cmp::min(max_tokens, available_tokens);
        warn!("Adjusted chat max_tokens from {} to {} to fit context window (estimated prompt: {} tokens)", 
              max_tokens, adjusted_max_tokens, estimated_prompt_tokens);
        adjusted_max_tokens
    } else {
        max_tokens
    };
    let top_k = req.options.as_ref().and_then(|o| o.top_k).unwrap_or(40);
    let top_p = req.options.as_ref().and_then(|o| o.top_p).unwrap_or(0.9);
    let temperature = req.options.as_ref().and_then(|o| o.temperature).unwrap_or(0.8);
    let repeat_penalty = req.options.as_ref().and_then(|o| o.repeat_penalty).unwrap_or(1.1);
    let max_context_len = state.max_context_len;

    info!("Chat parameters: max_tokens={}, top_k={}, top_p={:.2}, temperature={:.2}, repeat_penalty={:.2}",
           max_tokens, top_k, top_p, temperature, repeat_penalty);
    debug!("Starting RKLLM chat inference with params: max_tokens={}, top_k={}, top_p={}, temperature={}",
           max_tokens, top_k, top_p, temperature);

    let result = tokio::task::spawn_blocking(move || {
        let start_time = std::time::Instant::now();
        debug!("Initializing RKLLM model for chat...");
        // Create parameters inside the blocking task
        let model_path_cstr = std::ffi::CString::new(model_path)?;
        let mut param = RKLLMParam {
            model_path: model_path_cstr.as_ptr() as *const std::os::raw::c_char,
            max_context_len: max_context_len,
            max_new_tokens: max_tokens,
            top_k: top_k,
            top_p: top_p,
            temperature: temperature,
            repeat_penalty: repeat_penalty,
            ..Default::default()
        };

        let infer_params = RKLLMInferParam {
            mode: RKLLMInferMode::InferGenerate,
            lora_params: None,
            prompt_cache_params: None,
            keep_history: KeepHistory::NoKeepHistory,
        };

        let load_start = std::time::Instant::now();
        let llm_handle = match rkllm_init(&mut param) {
            Ok(handle) => {
                info!("RKLLM model initialized successfully for chat");
                handle
            }
            Err(e) => {
                error!("Failed to initialize RKLLM model for chat: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("RKLLM init failed: {:?}", e)));
            }
        };
        let load_duration = load_start.elapsed().as_nanos() as u64;

        let input = RKLLMInput {
            input_type: RKLLMInputType::Prompt(prompt.clone()),
            enable_thinking: false,
            role: RKLLMInputRole::User,
        };

        let prompt_eval_start = std::time::Instant::now();
        debug!("Running RKLLM chat inference...");
        let run_result = llm_handle.run(
            input,
            Some(infer_params),
            callback,
        );
        let prompt_eval_duration = prompt_eval_start.elapsed().as_nanos() as u64;

        match run_result {
            Ok(_) => info!("RKLLM chat inference completed successfully"),
            Err(e) => {
                error!("RKLLM chat inference failed: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("RKLLM run failed: {:?}", e)));
            }
        }

        match llm_handle.destroy() {
            Ok(_) => debug!("RKLLM handle destroyed successfully for chat"),
            Err(e) => warn!("Failed to destroy RKLLM handle for chat: {:?}", e),
        }

        let total_duration = start_time.elapsed().as_nanos() as u64;
        
        // Estimate token counts (rough approximation)
        let prompt_eval_count = (prompt.len() / 4) as u32; // Rough estimate: ~4 chars per token
        let response_len = response_text.blocking_lock().len();
        let eval_count = (response_len / 4) as u32; // Rough estimate
        let eval_duration = total_duration - load_duration - prompt_eval_duration;

        Ok((response_text.blocking_lock().clone(), load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration))
    }).await;

    match result {
        Ok(Ok((response, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration))) => {
            // Use the model's response as-is, without any content manipulation
            let response_content = response.trim().to_string();

            // Only handle truly empty responses
            let final_response_content = if response_content.is_empty() {
                warn!("Generated response was empty, using fallback message");
                "I apologize, but I couldn't generate a response. Please try again.".to_string()
            } else {
                response_content
            };

            let response_preview = if final_response_content.len() > 100 { 
                format!("{}...", &final_response_content[..100]) 
            } else { 
                final_response_content.clone() 
            };

            info!("Chat request completed successfully!");
            info!("Response stats: length={} chars, estimated_tokens={}", final_response_content.len(), eval_count);
            info!("Timing: total={:.2}ms, load={:.2}ms, prompt_eval={:.2}ms, generation={:.2}ms", 
                  total_duration as f64 / 1_000_000.0,
                  load_duration as f64 / 1_000_000.0,
                  prompt_eval_duration as f64 / 1_000_000.0,
                  eval_duration as f64 / 1_000_000.0);
            debug!("Generated chat response preview: {}", response_preview);
            info!("=== CHAT REQUEST END ===");

            Ok(Json(ChatResponse {
                model: req.model,
                created_at: chrono::Utc::now().to_rfc3339(),
                message: Message {
                    role: "assistant".to_string(),
                    content: final_response_content,
                },
                done: true,
                done_reason: Some("stop".to_string()),
                total_duration: Some(total_duration),
                load_duration: Some(load_duration),
                prompt_eval_count: Some(prompt_eval_count),
                prompt_eval_duration: Some(prompt_eval_duration),
                eval_count: Some(eval_count),
                eval_duration: Some(eval_duration),
            }))
        }
        Ok(Err(e)) => {
            error!("=== CHAT REQUEST FAILED ===");
            error!("Chat inference task failed: {}", e);
            error!("Model: {}, Messages: {}", req.model, req.messages.len());
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(e) => {
            error!("=== CHAT REQUEST FAILED ===");
            error!("Chat task spawn failed: {}", e);
            error!("Model: {}, Messages: {}", req.model, req.messages.len());
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn version_handler() -> Json<VersionResponse> {
    info!("Received version request");
    debug!("Returning server version: 0.9.6");
    Json(VersionResponse {
        version: "0.9.6".to_string(),
    })
}

async fn tags_handler(State(state): State<AppState>) -> Json<TagsResponse> {
    info!("Received tags request - listing available models");
    
    // Dynamically list available .rkllm models from the search paths
    let available_models = list_available_models(&state.model_path);
    
    let models: Vec<ModelInfo> = available_models.iter().map(|model_name| {
        // Try to get file size if possible
        let size = resolve_model_path(model_name, &state.model_path)
            .ok()
            .and_then(|path| std::fs::metadata(&path).ok())
            .map(|meta| meta.len())
            .unwrap_or(0);
        
        ModelInfo {
            name: model_name.clone(),
            model: model_name.clone(),
            modified_at: chrono::Utc::now().to_rfc3339(),
            size,
            digest: format!("sha256:{:064x}", model_name.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64))),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "rkllm".to_string(),
                family: "rkllm".to_string(),
                families: Some(vec!["rkllm".to_string()]),
                parameter_size: "unknown".to_string(),
                quantization_level: "unknown".to_string(),
            },
        }
    }).collect();
    
    let response = TagsResponse { models };
    info!("Returning {} available model(s)", response.models.len());
    debug!("Available models: {:?}", response.models.iter().map(|m| &m.name).collect::<Vec<_>>());
    Json(response)
}

async fn show_handler(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> Result<Json<ShowResponse>, StatusCode> {
    info!("Received show request for model: '{}'", req.model);
    debug!("Processing model information request for: {}", req.model);
    
    // Try to resolve the model path to verify it exists
    match resolve_model_path(&req.model, &state.model_path) {
        Ok(path) => {
            info!("Model '{}' found at {:?}, returning detailed information", req.model, path);
            
            // Get file size
            let size = std::fs::metadata(&path)
                .map(|meta| meta.len())
                .unwrap_or(0);
            
            let response = ShowResponse {
                modelfile: format!("FROM {}\nPARAMETER temperature 0.8\nPARAMETER top_p 0.9", req.model),
                parameters: "temperature 0.8\ntop_p 0.9\ntop_k 40\nrepeat_penalty 1.1".to_string(),
                template: "{{ if .System }}System: {{ .System }}\n\n{{ end }}{{ if .Prompt }}User: {{ .Prompt }}\n\nAssistant: {{ end }}".to_string(),
                details: ModelDetails {
                    parent_model: "".to_string(),
                    format: "rkllm".to_string(),
                    family: "rkllm".to_string(),
                    families: Some(vec!["rkllm".to_string()]),
                    parameter_size: "unknown".to_string(),
                    quantization_level: "unknown".to_string(),
                },
                model_info: ModelInfo {
                    name: req.model.clone(),
                    model: req.model.clone(),
                    modified_at: chrono::Utc::now().to_rfc3339(),
                    size,
                    digest: format!("sha256:{:064x}", req.model.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64))),
                    details: ModelDetails {
                        parent_model: "".to_string(),
                        format: "rkllm".to_string(),
                        family: "rkllm".to_string(),
                        families: Some(vec!["rkllm".to_string()]),
                        parameter_size: "unknown".to_string(),
                        quantization_level: "unknown".to_string(),
                    },
                },
                capabilities: vec!["completion".to_string(), "chat".to_string()],
            };
            
            debug!("Model info response prepared for '{}' - size: {} bytes", req.model, response.model_info.size);
            Ok(Json(response))
        }
        Err(e) => {
            warn!("Model '{}' not found: {}", req.model, e);
            error!("Requested model '{}' not found", req.model);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

async fn running_models_handler(State(state): State<AppState>) -> Json<RunningModelsResponse> {
    info!("Received running models request");
    // Since we load models per-request, there are no persistently running models
    // But we can list available models as if they're ready to run
    let available_models = list_available_models(&state.model_path);
    
    let models: Vec<RunningModelInfo> = available_models.iter().filter_map(|model_name| {
        // Try to get file size if possible
        let path = resolve_model_path(model_name, &state.model_path).ok()?;
        let size = std::fs::metadata(&path).ok()?.len();
        
        Some(RunningModelInfo {
            name: model_name.clone(),
            model: model_name.clone(),
            size,
            digest: format!("sha256:{:064x}", model_name.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64))),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "rkllm".to_string(),
                family: "rkllm".to_string(),
                families: Some(vec!["rkllm".to_string()]),
                parameter_size: "unknown".to_string(),
                quantization_level: "unknown".to_string(),
            },
            expires_at: (chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339(),
            size_vram: size,
        })
    }).collect();
    
    let response = RunningModelsResponse { models };
    info!("Returning {} running model(s)", response.models.len());
    Json(response)
}

async fn openai_chat_handler(
    State(state): State<AppState>,
    Json(req): Json<OpenAIChatRequest>,
) -> Result<Json<OpenAIChatResponse>, StatusCode> {
    info!("=== OPENAI CHAT REQUEST START ===");
    info!("Received OpenAI chat request for model: '{}'", req.model);
    //info!("Chat details: {} messages, stream={:?}", req.messages.len(), req.stream);
    //debug!("OpenAI request: model='{}', temp={:?}, max_tokens={:?}, top_p={:?}", 
    //       req.model, req.temperature, req.max_tokens, req.top_p);
    
    // Log message summary
    for (i, message) in req.messages.iter().enumerate() {
        let content_preview = if message.content.len() > 50 { 
            format!("{}...", &message.content[..50]) 
        } else { 
            message.content.clone() 
        };
        debug!("Message {}: role='{}', content_length={}, preview='{}'", 
               i + 1, message.role, message.content.len(), content_preview);
    }

    // Convert OpenAI request to our internal format
    let ollama_messages: Vec<Message> = req.messages.iter().map(|msg| Message {
        role: msg.role.clone(),
        content: msg.content.clone(),
    }).collect();

    let options = GenerateOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        max_tokens: req.max_tokens,
        repeat_penalty: None,
    };

    let chat_request = ChatRequest {
        model: req.model.clone(),
        messages: ollama_messages.clone(),
        stream: req.stream,
        options: Some(options),
    };

    debug!("Converted to Ollama format: model='{}', {} messages", 
           chat_request.model, chat_request.messages.len());
    debug!("Calling internal chat_handler...");

    // Use existing chat handler logic
    match chat_handler(State(state), Json(chat_request)).await {
        Ok(Json(chat_response)) => {
            info!("OpenAI chat request completed successfully!");
            
            // Clean the response content to remove <think> tags
            let cleaned_content = clean_response_content(&chat_response.message.content);
            debug!("Cleaned response: removed thinking tags, final length: {} chars", cleaned_content.len());
            
            // Convert response to OpenAI format
            let openai_response = OpenAIChatResponse {
                id: format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', "")),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: req.model,
                choices: vec![OpenAIChoice {
                    index: 0,
                    message: OpenAIMessage {
                        role: chat_response.message.role,
                        content: cleaned_content,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: OpenAIUsage {
                    prompt_tokens: chat_response.prompt_eval_count.unwrap_or(0),
                    completion_tokens: chat_response.eval_count.unwrap_or(0),
                    total_tokens: chat_response.prompt_eval_count.unwrap_or(0) + chat_response.eval_count.unwrap_or(0),
                },
            };

            // Log the complete response being sent to VS Code
            debug!("Response being sent to VS Code:");
            debug!("  ID: {}", openai_response.id);
            debug!("  Object: {}", openai_response.object);
            debug!("  Created: {}", openai_response.created);
            debug!("  Model: {}", openai_response.model);
            debug!("  Choices count: {}", openai_response.choices.len());
            if !openai_response.choices.is_empty() {
                debug!("  Choice[0].index: {}", openai_response.choices[0].index);
                debug!("  Choice[0].message.role: {}", openai_response.choices[0].message.role);
                debug!("  Choice[0].message.content: '{}'", openai_response.choices[0].message.content);
                debug!("  Choice[0].finish_reason: {}", openai_response.choices[0].finish_reason);
            }
            debug!("  Usage.prompt_tokens: {}", openai_response.usage.prompt_tokens);
            debug!("  Usage.completion_tokens: {}", openai_response.usage.completion_tokens);
            debug!("  Usage.total_tokens: {}", openai_response.usage.total_tokens);
            
            info!("=== OPENAI CHAT REQUEST END ===");
            Ok(Json(openai_response))
        }
        Err(status) => {
            error!("=== OPENAI CHAT REQUEST FAILED ===");
            error!("Chat handler returned error: {:?}", status);
            error!("Request details: model='{}', {} messages", req.model, req.messages.len());
            for (i, msg) in req.messages.iter().enumerate() {
                error!("  Message {}: role='{}', content='{}'", i+1, msg.role, msg.content);
            }
            Err(status)
        }
    }
}

async fn not_found_handler(uri: axum::http::Uri) -> impl axum::response::IntoResponse {
    warn!("Client attempted to access non-existing endpoint: {}", uri);
    error!("Requested non-existing endpoint: {}", uri);
    debug!("Available endpoints: /api/version, /api/tags, /api/ps, /api/show, /api/generate, /api/chat, /v1/chat/completions");
    
    (
        axum::http::StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": format!("Endpoint '{}' not found", uri),
                "type": "not_found",
                "code": 404
            }
        }))
    )
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .without_time()
                .with_target(false)
        )
        .init();

    info!("🚀 Starting Platinenmachergpt server...");
    info!("Version: 1.0.0");
    info!("Build: Rust compilation");

    // Load environment variables from .env file
    dotenv().ok();
    info!("Environment variables loaded from .env file");

    // For now, we'll initialize the model per request
    // In production, you might want to load it once and reuse
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/path/to/your/model.rkllm".to_string());

    // Memory and context limits configuration
    let max_context_len: i32 = std::env::var("MAX_CONTEXT_LEN")
        .unwrap_or_else(|_| "512".to_string())
        .parse()
        .unwrap_or(512);
    
    let max_new_tokens_default: i32 = std::env::var("MAX_NEW_TOKENS_DEFAULT")
        .unwrap_or_else(|_| "128".to_string())
        .parse()
        .unwrap_or(128);
    
    let max_new_tokens_limit: i32 = std::env::var("MAX_NEW_TOKENS_LIMIT")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .unwrap_or(256);
    
    let max_prompt_length: usize = std::env::var("MAX_PROMPT_LENGTH")
        .unwrap_or_else(|_| "2048".to_string())
        .parse()
        .unwrap_or(2048);

    info!("📁 Using model path: {}", model_path);
    info!("🧠 Memory limits: max_context_len={}, max_new_tokens_default={}, max_new_tokens_limit={}, max_prompt_length={}", 
          max_context_len, max_new_tokens_default, max_new_tokens_limit, max_prompt_length);
    debug!("Model path source: {}", if std::env::var("MODEL_PATH").is_ok() { 
        "environment variable" 
    } else { 
        "default fallback" 
    });

    let state = AppState {
        llm_handle: Arc::new(Mutex::new(None)),
        model_path: model_path.clone(),
        max_context_len,
        max_new_tokens_default,
        max_new_tokens_limit,
        max_prompt_length,
    };

    info!("🔧 Configuring API endpoints...");
    let app = Router::new()
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/api/ps", get(running_models_handler))
        .route("/v1/chat/completions", post(openai_chat_handler))
        .layer(CorsLayer::permissive())
        .fallback(not_found_handler)
        .with_state(state);

    info!("📡 Available endpoints:");
    info!("  GET  /api/version   - Server version information");
    info!("  GET  /api/tags      - List available models");
    info!("  GET  /api/ps        - List running models");
    info!("  POST /api/show      - Show detailed model information");
    info!("  POST /api/generate  - Text generation");
    info!("  POST /api/chat      - Chat completion");
    info!("  POST /v1/chat/completions - OpenAI-compatible chat completion");

    let bind_address = "0.0.0.0:11434";
    info!("🌐 Binding to address: {}", bind_address);
    
    let listener = match tokio::net::TcpListener::bind(bind_address).await {
        Ok(listener) => {
            info!("✅ Successfully bound to {}", bind_address);
            listener
        }
        Err(e) => {
            error!("❌ Failed to bind to {}: {}", bind_address, e);
            std::process::exit(1);
        }
    };

    info!("🚀 Platinenmachergpt server listening on http://{}", bind_address);
    info!("📋 Server ready to accept requests!");
    info!("=== SERVER STARTUP COMPLETE ===");
    
    if let Err(e) = axum::serve(listener, app).await {
        error!("❌ Server error: {}", e);
        std::process::exit(1);
    }
}
