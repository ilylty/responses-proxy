//! `POST /v1/responses/input_tokens` — token counting endpoint.

use crate::convert::responses_to_chat;
use crate::types::chat;
use crate::types::responses::{Error, Request};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

/// POST /v1/responses/input_tokens — count tokens without generating.
///
/// Converts the request to Chat API messages and estimates token count.
/// Uses a simple chars/4 heuristic (common approximation for English text).
pub async fn input_tokens(
    State(state): State<crate::app::State>,
    Json(req): Json<Request>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let chat_req = responses_to_chat(req, &state)
        .await
        .map_err(|unsupported| {
            let err =
                Error::invalid_request(format!("Unsupported features: {}", unsupported.join(", ")));
            (StatusCode::BAD_REQUEST, Json(err.to_http_json()))
        })?;

    let estimated = estimate_tokens(&chat_req);

    Ok(Json(serde_json::json!({
        "object": "response.input_tokens",
        "input_tokens": estimated,
    })))
}

/// Estimate token count from a Chat API request.
/// Falls back to chars/4 when no tokenizer is available.
fn estimate_tokens(req: &chat::Request) -> i64 {
    let mut total_chars: usize = 0;

    for msg in &req.messages {
        total_chars += count_message_chars(msg);
    }

    // Rough heuristic: ~4 characters per token for English text
    if total_chars == 0 {
        0
    } else {
        (total_chars / 4).max(1) as i64
    }
}

fn count_message_chars(msg: &chat::MessageRequest) -> usize {
    match msg {
        chat::MessageRequest::System(m) => count_msg_content(&m.content),
        chat::MessageRequest::Developer(m) => count_msg_content(&m.content),
        chat::MessageRequest::User(m) => count_user_content(&m.content),
        chat::MessageRequest::Assistant(m) => match &m.content {
            Some(c) => count_assistant_content(c),
            None => 0,
        },
        chat::MessageRequest::Tool(m) => count_msg_content(&m.content),
        chat::MessageRequest::Function(m) => m.content.len(),
    }
}

fn count_msg_content(c: &chat::MessageContent) -> usize {
    match c {
        chat::MessageContent::Text(s) => s.len(),
        chat::MessageContent::Parts(parts) => parts.iter().map(|p| p.text.len()).sum(),
    }
}

fn count_user_content(c: &chat::UserContent) -> usize {
    match c {
        chat::UserContent::Text(s) => s.len(),
        chat::UserContent::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                chat::ContentPart::Text { text } => text.len(),
                chat::ContentPart::Image { .. } => 85,
                chat::ContentPart::File { .. } => 0,
                chat::ContentPart::Audio { .. } => 0,
                chat::ContentPart::Refusal { .. } => 0,
            })
            .sum(),
    }
}

fn count_assistant_content(c: &chat::AssistantContent) -> usize {
    match c {
        chat::AssistantContent::Text(s) => s.len(),
        chat::AssistantContent::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                chat::ContentPart::Text { text } => text.len(),
                chat::ContentPart::Refusal { .. } => 0,
                _ => 0,
            })
            .sum(),
    }
}
