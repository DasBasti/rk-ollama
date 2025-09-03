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

#[derive(Deserialize, Serialize, Debug)]
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
    name: String,
}

#[derive(Serialize)]
struct ShowResponse {
    modelfile: String,
    parameters: String,
    template: String,
    details: ModelDetails,
    model_info: ModelInfo,
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

    let response_text = Arc::new(Mutex::new(String::new()));
    let callback = ApiCallbackHandler {
        response: Arc::clone(&response_text),
    };

    // Run inference in a blocking task
    let model_path = state.model_path.clone();
    let prompt = req.prompt.clone();
    let max_tokens = req.options.as_ref().and_then(|o| o.max_tokens).unwrap_or(256);
    let top_k = req.options.as_ref().and_then(|o| o.top_k).unwrap_or(40);
    let top_p = req.options.as_ref().and_then(|o| o.top_p).unwrap_or(0.9);
    let temperature = req.options.as_ref().and_then(|o| o.temperature).unwrap_or(0.8);
    let repeat_penalty = req.options.as_ref().and_then(|o| o.repeat_penalty).unwrap_or(1.1);

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
            max_context_len: 1024,
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

    let response_text = Arc::new(Mutex::new(String::new()));
    let callback = ApiCallbackHandler {
        response: Arc::clone(&response_text),
    };

    // Run inference in a blocking task
    let model_path = state.model_path.clone();
    let max_tokens = req.options.as_ref().and_then(|o| o.max_tokens).unwrap_or(256);
    let top_k = req.options.as_ref().and_then(|o| o.top_k).unwrap_or(40);
    let top_p = req.options.as_ref().and_then(|o| o.top_p).unwrap_or(0.9);
    let temperature = req.options.as_ref().and_then(|o| o.temperature).unwrap_or(0.8);
    let repeat_penalty = req.options.as_ref().and_then(|o| o.repeat_penalty).unwrap_or(1.1);

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
            max_context_len: 1024,
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
    debug!("Returning server version: 1.0.0");
    Json(VersionResponse {
        version: "1.0.0".to_string(),
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
            digest: "sha256:887827d6fc84bb81b9b4c64d3aae7e9c8b9e8a5f8c3d7a8b5e6f9c3d7a8b5e6f".to_string(),
            details: ModelDetails {
                parent_model: "".to_string(),
                format: "gguf".to_string(),
                family: "gemma".to_string(),
                families: Some(vec!["gemma".to_string()]),
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
    info!("Received show request for model: '{}'", req.name);
    debug!("Processing model information request for: {}", req.name);
    
    // For now, return information about our single model
    // In a real implementation, you'd look up the specific model
    if req.name == "gemma2:2b" || req.name.starts_with("gemma") {
        info!("Model '{}' found, returning detailed information", req.name);
        debug!("Generating modelfile and template information for: {}", req.name);
        
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
                name: req.name.clone(),
                model: req.name.clone(),
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
        };
        
        debug!("Model info response prepared for '{}' - size: {} bytes", req.name, response.model_info.size);
        Ok(Json(response))
    } else {
        warn!("Model '{}' not found in available models", req.name);
        error!("Requested model '{}' not found", req.name);
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

async fn not_found_handler(uri: axum::http::Uri) -> impl axum::response::IntoResponse {
    warn!("Client attempted to access non-existing endpoint: {}", uri);
    error!("Requested non-existing endpoint: {}", uri);
    debug!("Available endpoints: /api/version, /api/tags, /api/ps, /api/show, /api/generate, /api/chat");
    
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
                .unwrap_or_else(|_| "platinenmachergpt=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
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

    info!("📁 Using model path: {}", model_path);
    debug!("Model path source: {}", if std::env::var("MODEL_PATH").is_ok() { 
        "environment variable" 
    } else { 
        "default fallback" 
    });

    let state = AppState {
        llm_handle: Arc::new(Mutex::new(None)),
        model_path: model_path.clone(),
    };

    info!("🔧 Configuring API endpoints...");
    let app = Router::new()
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/api/ps", get(running_models_handler))
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
