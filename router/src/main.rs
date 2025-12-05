use axum::{
    extract::State,
    routing::{post},
    http::StatusCode,
    Json, Router,
};
mod models;
use crate::models::{
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    Message, Choice
};

use tonic::transport::Channel;
pub mod textdiffusion {
    pub mod v1 {
        tonic::include_proto!("textdiffusion.v1");
    }
}

use textdiffusion::v1::{
    text_generation_service_client::TextGenerationServiceClient,
    GenerateRequest,
};

#[derive(Clone)]
struct AppState {
    client: TextGenerationServiceClient<Channel>,
}

#[tokio::main]
async fn main()  -> anyhow::Result<()> {
    let channel: Channel = Channel::from_static("http://127.0.0.1:50057")
        .connect()
        .await?;

    let client = TextGenerationServiceClient::new(channel);

    let state = AppState { client };

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("localhost:3001").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>
) -> Result<Json<ChatCompletionResponse>, StatusCode>{
    println!("Handling chat completions...");

    let formatted_chat = format_chat(payload.messages).unwrap();

    let mut client = state.client.clone();

    let request = GenerateRequest {
        prompt: formatted_chat,
        max_output_tokens: 128,
        num_steps: 128,
        seed: 0,
        mask_id: 126336,
        block_length: 32,
        temperature: 0.0,
        request_id: "rust-test-1".to_string(),
    };

    let response = client.generate(request)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .into_inner();

    let cleaned_output = clean_assistant_response(&response.output_text);
    println!("{}", cleaned_output);

    let return_message = Message {
        role: "assistant".to_string(),
        content: cleaned_output,
    };
    let choice = Choice {
        index: 0,
        message: return_message,
        finish_reason: "stop".to_string(),
    };
    let chat_completion_response = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        choices: vec![choice],
    };

    Ok(Json(chat_completion_response))
}


// chat template taken from here -> https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/tokenizer_config.json
fn format_chat(messages: Vec<Message>) -> anyhow::Result<String> {
    let mut out = String::new();

    for (i, msg) in messages.iter().enumerate() {
        if i == 0 {
            // bos_token content from tokenizer_config: "<|startoftext|>"
            out.push_str("<|startoftext|>");
        }

        out.push_str("<|start_header_id|>");
        out.push_str(&msg.role);
        out.push_str("<|end_header_id|>\n\n");

        // trim like the template: message['content'] | trim
        out.push_str(msg.content.trim());

        out.push_str("<|eot_id|>");
    }

    // final generation prompt for assistant:
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    Ok(out)
}

fn clean_assistant_response(raw: &str) -> String {
    const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>";
    const END_MARKERS: [&str; 2] = ["<|eot_id|>", "<|endoftext|>"];

    let after_prefix = raw
        .rfind(ASSISTANT_PREFIX)
        .map(|idx| &raw[idx + ASSISTANT_PREFIX.len()..])
        .unwrap_or(raw)
        .trim_start();

    let cut_at = END_MARKERS
        .iter()
        .filter_map(|marker| after_prefix.find(marker))
        .min()
        .unwrap_or(after_prefix.len());

    after_prefix[..cut_at].trim().to_string()
}