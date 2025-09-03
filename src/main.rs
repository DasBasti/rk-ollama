use axum::{
    routing::post,
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
    info!("Received generate request for model: {}", req.model);
    debug!("Request details: prompt_length={}, stream={:?}", req.prompt.len(), req.stream);

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
            info!("Generate request completed successfully, response length: {}", response.len());
            debug!("Generated response: {}", response);
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
            error!("Inference task failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(e) => {
            error!("Task spawn failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[instrument(skip(state))]
async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    info!("Received chat request for model: {}", req.model);
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
    debug!("Converted chat to prompt: {}", prompt);

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
                response_content = response_content[assistant_pos + 10..].trim().to_string();
            }

            // If response is empty, provide a fallback
            if response_content.is_empty() {
                response_content = "I apologize, but I couldn't generate a response. Please try again.".to_string();
            }

            info!("Chat request completed successfully, response length: {}", response_content.len());
            debug!("Generated chat response: {}", response_content);

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
            error!("Chat inference task failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(e) => {
            error!("Chat task spawn failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
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

    info!("Starting Platinenmachergpt server...");

    // Load environment variables from .env file
    dotenv().ok();

    // For now, we'll initialize the model per request
    // In production, you might want to load it once and reuse
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/path/to/your/model.rkllm".to_string());

    info!("Using model path: {}", model_path);

    let state = AppState {
        llm_handle: Arc::new(Mutex::new(None)),
        model_path,
    };

    let app = Router::new()
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:11434").await.unwrap();
    info!("🚀 Platinenmachergpt server listening on http://0.0.0.0:11434");
    axum::serve(listener, app).await.unwrap();
}
