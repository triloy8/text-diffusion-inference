use axum::{
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

#[tokio::main]
async fn main(){
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions));

    let listener = tokio::net::TcpListener::bind("localhost:3001").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat_completions(
    Json(payload): Json<ChatCompletionRequest>
) -> (StatusCode, Json<ChatCompletionResponse>){
    println!("handling chat completions...");
    let return_message = Message {
        role: "assistant".to_string(),
        content: "blue".to_string(),
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
    
    (StatusCode::OK, Json(chat_completion_response))
}

