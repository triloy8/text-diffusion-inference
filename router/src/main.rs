use anyhow::Context;
use axum::{
    extract::State,
    http::{StatusCode, Uri},
    routing::post,
    Json, Router,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
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
mod tokenizer;
use crate::tokenizer::Tokenizer;

pub mod textdiffusion {
    pub mod v1 {
        tonic::include_proto!("textdiffusion.v1");
    }
}

use textdiffusion::v1::{
    text_generation_service_client::TextGenerationServiceClient,
    GenerateRequest,
    Tokens,
};

#[derive(Clone)]
struct AppState {
    client: TextGenerationServiceClient<Channel>,
    tokenizer: Tokenizer,
    mask_id: u32,
}

const DEFAULT_MAX_TOKENS: u32 = 128;
const DEFAULT_NUM_STEPS: u32 = 128;
const DEFAULT_SEED: u64 = 0;
const DEFAULT_MASK_ID: u32 = 126336;
const DEFAULT_BLOCK_LENGTH: u32 = 32;
const DEFAULT_TEMPERATURE: f32 = 0.0;

#[derive(Parser, Debug)]
#[command(name = "router", about = "Text Diffusion router")]
struct Cli {
    #[arg(long, default_value = "localhost")]
    host: String,
    #[arg(long, default_value_t = 3001)]
    port: u16,
    #[arg(long = "uds-path", value_name = "SOCKET")]
    uds_path: PathBuf,
    #[arg(long = "tokenizer-repo-id", value_name = "REPO")]
    tokenizer_repo_id: String,
    #[arg(long = "vocab-path", value_name = "vocab_path")]
    vocab_path: PathBuf,
    #[arg(long = "merges-filepath", value_name = "merges_filepath")]
    merges_filepath: PathBuf,
    #[arg(long = "specialtokens-filepath", value_name = "specialtokens_filepath")]
    specialtokens_filepath: PathBuf,
    #[arg(long = "mask-id", value_name = "MASK_ID", default_value_t = DEFAULT_MASK_ID)]
    mask_id: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let channel = connect_worker(&cli.uds_path).await?;

    let client = TextGenerationServiceClient::new(channel);

    let (vocab_path, merges_path, specialtokens_path) = download_tokenizer_files(
        &cli.tokenizer_repo_id,
        &cli.vocab_path,
        &cli.merges_filepath,
        &cli.specialtokens_filepath,
    )?;

    let tokenizer = Tokenizer::from_files(&vocab_path, &merges_path, &specialtokens_path)
        .map_err(anyhow::Error::new)?;

    let state = AppState { 
        client,
        tokenizer,
        mask_id: cli.mask_id,
    };

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
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    println!("handling chat completions...");

    let ChatCompletionRequest {
        model: _,
        messages,
        conversation_id,
        max_tokens,
        num_steps,
        seed,
        block_length,
        temperature,
    } = payload;

    let formatted_chat = format_chat(messages).unwrap();

    let mut client = state.client.clone();

    let max_output_tokens = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as i32;
    let steps = num_steps.unwrap_or(DEFAULT_NUM_STEPS) as i32;
    let block_len = block_length.unwrap_or(DEFAULT_BLOCK_LENGTH) as i32;
    let seed = seed.unwrap_or(DEFAULT_SEED);
    let mask_id = state.mask_id;
    let temperature = temperature.unwrap_or(DEFAULT_TEMPERATURE);

    let proto_ids = state.tokenizer.encode(formatted_chat).unwrap().into_iter().map(|id| id as u32).collect();
    let prompt_tokens = Tokens{ ids: proto_ids };

    let request = GenerateRequest {
        prompt_tokens: Some(prompt_tokens),
        max_output_tokens: max_output_tokens,
        num_steps: steps,
        seed: seed,
        mask_id: mask_id,
        block_length: block_len,
        temperature: temperature,
        request_id: conversation_id,
    };

    let response = client.generate(request)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .into_inner();

    let proto_ids = response.output_tokens.unwrap().ids.into_iter().map(|id| id as usize).collect();
    let output_text = state.tokenizer.decode(proto_ids).unwrap();

    let cleaned_output = clean_assistant_response(&output_text);

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

fn download_tokenizer_files(
    repo_id: &str,
    vocab: &Path,
    merges: &Path,
    special_tokens: &Path,
) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let api = Api::new().context("failed to initialize Hugging Face Hub API")?;
    let repo = api.repo(Repo::new(repo_id.to_owned(), RepoType::Model));

    let vocab_name = vocab.to_string_lossy().into_owned();
    let merges_name = merges.to_string_lossy().into_owned();
    let special_tokens_name = special_tokens.to_string_lossy().into_owned();

    let vocab_path = repo
        .get(&vocab_name)
        .with_context(|| format!("failed to download {vocab_name} from {repo_id}"))?;
    let merges_path = repo
        .get(&merges_name)
        .with_context(|| format!("failed to download {merges_name} from {repo_id}"))?;
    let special_tokens_path = repo
        .get(&special_tokens_name)
        .with_context(|| format!("failed to download {special_tokens_name} from {repo_id}"))?;

    Ok((vocab_path, merges_path, special_tokens_path))
}
