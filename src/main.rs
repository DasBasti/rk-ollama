use axum::{
    routing::{post, get},
    Router,
    Json,
    extract::State,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use rkllm_rs::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use dotenv::dotenv;
use tracing::{info, warn, error, debug, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

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

#[instrument(skip(state))]
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
    let model_path = state.model_path.clone();
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

#[instrument(skip(state))]
async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    info!("=== CHAT REQUEST START ===");
    info!("Received chat request for model: '{}'", req.model);
    info!("Chat details: {} messages, stream={:?}", req.messages.len(), req.stream);
    
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

    // Convert chat messages to a single prompt
    let mut prompt_parts = Vec::new();

    for message in &req.messages {
        match message.role.as_str() {
            "system" => {
                prompt_parts.push(format!("System: {}", message.content));
            }
            "user" => {
                prompt_parts.push(format!("User: {}", message.content));
            }
            "assistant" => {
                prompt_parts.push(format!("Assistant: {}", message.content));
            }
            _ => {
                debug!("Unknown message role: {}", message.role);
            }
        }
    }

    // Add the assistant prompt
    prompt_parts.push("Assistant:".to_string());

    let prompt = prompt_parts.join("\n\n");
    info!("Converted {} messages to prompt (length: {} chars)", req.messages.len(), prompt.len());
    debug!("Converted chat to prompt: {}", if prompt.len() > 200 { 
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
    let model_path = state.model_path.clone();
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
            let mut response_content = response;

            // Clean up the response (remove any system/user prefixes if they appear)
            if let Some(assistant_pos) = response_content.find("Assistant:") {
                debug!("Cleaning up response: removing 'Assistant:' prefix");
                response_content = response_content[assistant_pos + 10..].trim().to_string();
            }

            // If response is empty, provide a fallback
            if response_content.is_empty() {
                warn!("Generated response was empty, using fallback message");
                response_content = "I apologize, but I couldn't generate a response. Please try again.".to_string();
            }

            let response_preview = if response_content.len() > 100 { 
                format!("{}...", &response_content[..100]) 
            } else { 
                response_content.clone() 
            };

            info!("Chat request completed successfully!");
            info!("Response stats: length={} chars, estimated_tokens={}", response_content.len(), eval_count);
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
                    content: response_content,
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

async fn tags_handler() -> Json<TagsResponse> {
    info!("Received tags request - listing available models");
    // Return information about the available model
    // In a real implementation, you might scan a models directory or maintain a registry
    let response = TagsResponse {
        models: vec![ModelInfo {
            name: "gemma2:2b".to_string(),
            model: "gemma2:2b".to_string(),
            modified_at: chrono::Utc::now().to_rfc3339(),
            size: 1_073_741_824, // 1GB in bytes
            digest: "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf".to_string(),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "gemma3".to_string(),
                families: Some(vec!["gemma3".to_string()]),
                parameter_size: "2B".to_string(),
                quantization_level: "Q4_K_M".to_string(),
            },
        }],
    };
    info!("Returning {} available model(s)", response.models.len());
    debug!("Available models: {:?}", response.models.iter().map(|m| &m.name).collect::<Vec<_>>());
    Json(response)
}

async fn show_handler(Json(req): Json<ShowRequest>) -> Result<Json<ShowResponse>, StatusCode> {
    info!("Received show request for model: '{}'", req.model);
    debug!("Processing model information request for: {}", req.model);
    
    // For now, return information about our single model
    // In a real implementation, you'd look up the specific model
    if req.model == "gemma2:2b" || req.model.starts_with("gemma") {
        info!("Model '{}' found, returning detailed information", req.model);
        debug!("Generating modelfile and template information for: {}", req.model);
        
        let response = ShowResponse {
            modelfile: "FROM gemma2:2b\nPARAMETER temperature 0.8\nPARAMETER top_p 0.9".to_string(),
            parameters: "temperature 0.8\ntop_p 0.9\ntop_k 40\nrepeat_penalty 1.1".to_string(),
            template: "{{ if .System }}System: {{ .System }}\n\n{{ end }}{{ if .Prompt }}User: {{ .Prompt }}\n\nAssistant: {{ end }}".to_string(),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "gemma".to_string(),
                families: Some(vec!["gemma".to_string()]),
                parameter_size: "2B".to_string(),
                quantization_level: "Q4_K_M".to_string(),
            },
            model_info: ModelInfo {
                name: req.model.clone(),
                model: req.model.clone(),
                modified_at: chrono::Utc::now().to_rfc3339(),
                size: 1_073_741_824,
                digest: "sha256:887827d6fc84bb81b9b4c64d3aae7e9c8b9e8a5f8c3d7a8b5e6f9c3d7a8b5e6f".to_string(),
                details: ModelDetails {
                    parent_model: "".to_string(),
                    format: "gguf".to_string(),
                    family: "gemma".to_string(),
                    families: Some(vec!["gemma".to_string()]),
                    parameter_size: "2B".to_string(),
                    quantization_level: "Q4_K_M".to_string(),
                },
            },
            capabilities: vec!["completion".to_string(), "chat".to_string()],
        };
        
        debug!("Model info response prepared for '{}' - size: {} bytes", req.model, response.model_info.size);
        Ok(Json(response))
    } else {
        warn!("Model '{}' not found in available models", req.model);
        error!("Requested model '{}' not found", req.model);
        Err(StatusCode::NOT_FOUND)
    }
}

async fn running_models_handler() -> Json<RunningModelsResponse> {
    info!("Received running models request");
    // In a real implementation, track which models are actually loaded in memory
    // For now, simulate that our model is running
    let response = RunningModelsResponse {
        models: vec![RunningModelInfo {
            name: "gemma2:2b".to_string(),
            model: "gemma2:2b".to_string(),
            size: 1_073_741_824,
            digest: "sha256:887827d6fc84bb81b9b4c64d3aae7e9c8b9e8a5f8c3d7a8b5e6f9c3d7a8b5e6f".to_string(),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "gemma".to_string(),
                families: Some(vec!["gemma".to_string()]),
                parameter_size: "2B".to_string(),
                quantization_level: "Q4_K_M".to_string(),
            },
            expires_at: (chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339(),
            size_vram: 1_073_741_824,
        }],
    };
    info!("Returning {} running model(s)", response.models.len());
    Json(response)
}

#[instrument(skip(state))]
async fn openai_chat_handler(
    State(state): State<AppState>,
    Json(req): Json<OpenAIChatRequest>,
) -> Result<Json<OpenAIChatResponse>, StatusCode> {
    info!("=== OPENAI CHAT REQUEST START ===");
    info!("Received OpenAI chat request for model: '{}'", req.model);
    info!("Chat details: {} messages, stream={:?}", req.messages.len(), req.stream);
    debug!("OpenAI request: model='{}', temp={:?}, max_tokens={:?}, top_p={:?}", 
           req.model, req.temperature, req.max_tokens, req.top_p);
    
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
                        content: chat_response.message.content,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: OpenAIUsage {
                    prompt_tokens: chat_response.prompt_eval_count.unwrap_or(0),
                    completion_tokens: chat_response.eval_count.unwrap_or(0),
                    total_tokens: chat_response.prompt_eval_count.unwrap_or(0) + chat_response.eval_count.unwrap_or(0),
                },
            };

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
        .layer(TraceLayer::new_for_http())
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
