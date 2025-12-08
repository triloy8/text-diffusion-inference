use axum::{
    extract::State,
    http::{StatusCode, Uri},
    routing::post,
    Json, Router,
};
use clap::Parser;
use std::path::{Path, PathBuf};
use tokio::net::UnixStream;
use hyper_util::rt::TokioIo;
use tonic::transport::{Channel, Endpoint};
use tower::service_fn;
mod models;
use crate::models::{
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    Message, Choice
};

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

#[derive(Parser, Debug)]
#[command(name = "router", about = "Text Diffusion router")]
struct Cli {
    #[arg(long, default_value = "localhost")]
    host: String,
    #[arg(long, default_value_t = 3001)]
    port: u16,
    #[arg(long = "uds-path", value_name = "SOCKET")]
    uds_path: PathBuf,
}

#[tokio::main]
async fn main()  -> anyhow::Result<()> {
    let cli = Cli::parse();

    let channel = connect_worker(&cli.uds_path).await?;

    let client = TextGenerationServiceClient::new(channel);

    let state = AppState { client };

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn connect_worker(uds_path: &Path) -> anyhow::Result<Channel> {
    let uds = uds_path.to_path_buf();
    let endpoint = Endpoint::try_from("http://[::]:50057")?;
    let channel = endpoint
        .connect_with_connector(service_fn(move |_: Uri| {
            let uds = uds.clone();
            async move {
                let stream = UnixStream::connect(uds).await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(channel)
}

async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>
) -> Result<Json<ChatCompletionResponse>, StatusCode>{
    println!("handling chat completions...");

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
