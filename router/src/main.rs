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
    let mut client = state.client.clone();

    let request = GenerateRequest {
        prompt: "hello".to_string(),
        max_output_tokens: 256,
        num_steps: 256,
        seed: 0,
        mask_id: 50257,
        block_length: 256,
        temperature: 0.1,
        request_id: "rust-test-1".to_string(),
    };

    let response = client.generate(request)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .into_inner();

    let return_message = Message {
        role: "assistant".to_string(),
        content: response.output_text,
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

