//! Handler for the `response.create` WebSocket event.

use crate::convert::responses_to_chat;
use crate::types::chat::{self, MessageRequest};
use crate::types::event;
use crate::types::event::StreamEvent;
use crate::types::responses::{Error, Request, Response, ResponseStatus};
use crate::types::streaming::*;
use crate::types::websocket;
use axum::extract::ws::{Message as WsMsg, WebSocket};
use futures::StreamExt;

/// Build a minimal Response for lifecycle events.  Full output / usage are populated later.
fn ws_response(rid: &str, model: &str, now: i64, status: ResponseStatus) -> Response {
    Response {
        id: rid.to_string(),
        model: model.to_string(),
        status,
        created_at: now,
        parallel_tool_calls: true,
        ..Default::default()
    }
}

/// Handle a `response.create` event: parse, forward to upstream, stream back results.
pub(super) async fn handle(state: &crate::app::State, socket: &mut WebSocket, mut req: Request) {
    tracing::debug!("input items {}", req.input.len());

    let provider = match state.config().models.get(&req.model) {
        Some(p) => p.clone(),
        None => {
            let ws_err = websocket::ErrorEvent::new(
                400,
                Error::TYPE_INVALID_REQUEST,
                "model_not_found",
                format!("Unknown model: {}", req.model),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    if !provider.rewrite.responses_in.is_empty() {
        let mut body = match serde_json::to_value(&req) {
            Ok(body) => body,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
        if let Err(message) =
            crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.responses_in)
        {
            let ws_err = websocket::ErrorEvent::new(
                500,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                message,
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
        req = match serde_json::from_value(body) {
            Ok(req) => req,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    400,
                    Error::TYPE_INVALID_REQUEST,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
    }

    let model = req.model.clone();
    let generate = req.generate;

    let rid = format!("resp_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let mid = format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));

    // Convert to Chat API (responses_to_chat handles history + instructions)
    let mut chat_req = match responses_to_chat(req, state).await {
        Ok(cr) => cr,
        Err(unsupported) => {
            let ws_err = websocket::ErrorEvent::new(
                400,
                Error::TYPE_INVALID_REQUEST,
                "unsupported_feature",
                format!("Unsupported features: {}", unsupported.join(", ")),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };
    chat_req.model = provider.model.clone();
    let mut full_input_messages = chat_req.messages.clone();

    // If generate=false, just echo lifecycle events without calling upstream
    if !generate {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let resp_in_progress = ws_response(&rid, &model, now, ResponseStatus::InProgress);
        let resp_completed = ws_response(&rid, &model, now, ResponseStatus::Completed);

        for event in [
            StreamEvent::Created(event::Created {
                response: resp_in_progress.clone(),
                sequence_number: 0,
            }),
            StreamEvent::InProgress(event::InProgress {
                response: resp_in_progress,
                sequence_number: 1,
            }),
            StreamEvent::Completed(event::Completed {
                response: resp_completed,
                sequence_number: 2,
            }),
        ] {
            match prepare_stream_event(event, &provider.rewrite.responses_out) {
                Ok(prepared) => {
                    super::send(socket, &prepared.body.to_string()).await;
                }
                Err(message) => {
                    let ws_err = websocket::ErrorEvent::new(
                        500,
                        Error::TYPE_SERVER_ERROR,
                        Error::CODE_SERVER_ERROR,
                        message,
                    );
                    super::send(socket, &ws_err.to_json_string()).await;
                    return;
                }
            }
        }

        state.store().put(rid, full_input_messages).await;
        return;
    }

    let url = format!("{}/chat/completions", provider.base_url);
    let request = state
        .http_client()
        .post(&url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json");
    let request = if provider.rewrite.chat_out.is_empty() {
        tracing::debug!(
            "chat request: {}",
            serde_json::to_string(&chat_req).unwrap_or("".to_string())
        );
        request.json(&chat_req)
    } else {
        let mut body = match serde_json::to_value(&chat_req) {
            Ok(body) => body,
            Err(e) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    e.to_string(),
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        };
        if let Err(message) = crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.chat_out) {
            let ws_err = websocket::ErrorEvent::new(
                500,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                message,
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
        tracing::debug!(
            "chat request: {}",
            serde_json::to_string(&body).unwrap_or("".to_string())
        );
        request.json(&body)
    };
    let stream_resp = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            let ws_err = websocket::ErrorEvent::new(
                502,
                Error::TYPE_SERVER_ERROR,
                Error::CODE_SERVER_ERROR,
                format!("Upstream error: {e}"),
            );
            super::send(socket, &ws_err.to_json_string()).await;
            return;
        }
    };

    if !stream_resp.status().is_success() {
        let status_code = stream_resp.status().as_u16();
        let ws_err = websocket::ErrorEvent::new(
            status_code,
            Error::TYPE_SERVER_ERROR,
            Error::CODE_SERVER_ERROR,
            format!(
                "Upstream error:  {}",
                stream_resp.text().await.unwrap_or("".into())
            ),
        );
        super::send(socket, &ws_err.to_json_string()).await;
        return;
    }

    // Send lifecycle start events (typed, with sequence_number)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let resp = ws_response(&rid, &model, now, ResponseStatus::InProgress);
    let mut initial_events = Vec::new();
    for event in [
        StreamEvent::Created(event::Created {
            response: resp.clone(),
            sequence_number: 0,
        }),
        StreamEvent::InProgress(event::InProgress {
            response: resp,
            sequence_number: 1,
        }),
    ] {
        match prepare_stream_event(event, &provider.rewrite.responses_out) {
            Ok(prepared) => {
                initial_events.push(prepared.event);
                super::send(socket, &prepared.body.to_string()).await;
            }
            Err(message) => {
                let ws_err = websocket::ErrorEvent::new(
                    500,
                    Error::TYPE_SERVER_ERROR,
                    Error::CODE_SERVER_ERROR,
                    message,
                );
                super::send(socket, &ws_err.to_json_string()).await;
                return;
            }
        }
    }

    // Register cancellation token for HTTP cancel endpoint
    let cancel_rx = state.store().register_cancel_token(&rid).await;

    // Stream loop: read SSE chunks + handle cancel
    let stream_context = WsStreamContext {
        rid: &rid,
        mid: &mid,
        model: &model,
        chat_in: &provider.rewrite.chat_in,
        responses_out: &provider.rewrite.responses_out,
        now,
        compact_key: state.compact_key(),
    };
    let (response_msg, cancelled, stream_events) =
        run_stream(socket, stream_resp, stream_context, cancel_rx).await;
    let events = initial_events;
    // stream_events are already sent, just used for counting
    let total_events = events.len() + stream_events.len();

    // Clean up cancel token (run_stream already handled the actual cancellation check)
    state.store().unregister_cancel_token(&rid).await;

    // Persist accumulated history
    if !cancelled {
        // Append assistant response to input messages and store
        let assistant_msg: MessageRequest = response_msg.into();
        let has_reasoning =
            matches!(&assistant_msg, MessageRequest::Assistant(a) if a.reasoning_content.is_some());
        tracing::info!(
            total_events,
            has_reasoning,
            msg_count = full_input_messages.len() + 1,
            "WS: storing history"
        );
        full_input_messages.push(assistant_msg);
        state.store().put(rid, full_input_messages).await;
    }
}

/// Relay SSE chunks from upstream to WebSocket, with cancel detection.
/// Returns (response_message, cancelled, collected_events).
struct WsStreamContext<'a> {
    rid: &'a str,
    mid: &'a str,
    model: &'a str,
    chat_in: &'a crate::config::RewriteConfig,
    responses_out: &'a crate::config::RewriteConfig,
    now: i64,
    compact_key: Option<&'a [u8; 32]>,
}

async fn run_stream(
    socket: &mut WebSocket,
    stream_resp: reqwest::Response,
    context: WsStreamContext<'_>,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) -> (chat::ResponseMessage, bool, Vec<StreamEvent>) {
    let mut buf = String::new();
    let mut ss = StreamState::new(
        context.rid.to_string(),
        context.mid.to_string(),
        context.model.to_string(),
    );
    ss.has_started = true;
    ss.created = context.now;
    ss.compact_key = context.compact_key.copied();
    let mut byte_stream = stream_resp.bytes_stream();
    let mut cancelled = false;
    let mut collected_events: Vec<StreamEvent> = Vec::new();

    loop {
        tokio::select! {
            _ = cancel_rx.changed() => {
                tracing::info!(response_id = %context.rid, "WS stream cancelled via HTTP");
                cancelled = true;
                break;
            }
            chunk = byte_stream.next() => {
                match chunk {
                    Some(Ok(b)) => {
                        buf.push_str(&String::from_utf8_lossy(&b));
                        while let Some(pos) = buf.find("\n\n") {
                            let ev = buf[..pos].trim().to_string();
                            buf = buf[pos + 2..].to_string();
                            if let Some(data) = ev.lines()
                                .find(|l| l.starts_with("data:"))
                                .and_then(|l| l.strip_prefix("data:").map(|s| s.trim()))
                            {
                                tracing::trace!(%data, "Chat API delta");
                                match process_upstream_stream_data(
                                    &mut ss,
                                    data,
                                    context.chat_in,
                                    context.responses_out,
                                ) {
                                    Ok(events) => {
                                        for prepared in events {
                                            if !prepared.event_type.ends_with("delta") {
                                                tracing::info!("WS event: {}", prepared.event_type);
                                                tracing::debug!("WS event details: {}", prepared.body);
                                            }
                                            let msg = prepared.body.to_string();
                                            tracing::debug!("WS send: {msg}");
                                            collected_events.push(prepared.event);
                                            if socket.send(WsMsg::Text(msg.into())).await.is_err() {
                                                tracing::info!("WS send failed");
                                                return (ss.to_response_message(), false, collected_events);
                                            }
                                        }
                                    }
                                    Err(message) => {
                                        let ws_err = websocket::ErrorEvent::new(
                                            500,
                                            Error::TYPE_SERVER_ERROR,
                                            Error::CODE_SERVER_ERROR,
                                            message,
                                        );
                                        let msg = ws_err.to_json_string();
                                        tracing::debug!("WS send: {msg}");
                                        let _ = socket.send(WsMsg::Text(msg.into())).await;
                                        return (ss.to_response_message(), false, collected_events);
                                    }
                                }
                            }
                        }
                    }
                    _ => break,
                }
            }
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(WsMsg::Text(t)))
                        if t.trim() == r#"{"type":"response.cancel"}"# =>
                    {
                        tracing::info!("WS cancel received during streaming");
                        cancelled = true;
                        break;
                    }
                    _ => break,
                }
            }
        }
    }
    (ss.to_response_message(), cancelled, collected_events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_response_builds_correctly() {
        let resp = ws_response("resp_1", "gpt-5", 1000, ResponseStatus::InProgress);
        assert_eq!(resp.id, "resp_1");
        assert_eq!(resp.model, "gpt-5");
        assert_eq!(resp.created_at, 1000);
        assert_eq!(resp.status, ResponseStatus::InProgress);
        assert!(resp.parallel_tool_calls);
        assert!(resp.output.is_empty());
    }

    #[test]
    fn ws_response_completed_status() {
        let resp = ws_response("resp_x", "claude", 500, ResponseStatus::Completed);
        assert_eq!(resp.status, ResponseStatus::Completed);
    }

    #[test]
    fn ws_response_all_statuses() {
        let statuses = &[
            ResponseStatus::Queued,
            ResponseStatus::InProgress,
            ResponseStatus::Completed,
            ResponseStatus::Failed,
            ResponseStatus::Incomplete,
            ResponseStatus::Cancelled,
        ];
        for st in statuses {
            let resp = ws_response("r", "m", 0, *st);
            assert_eq!(resp.id, "r");
        }
    }
}
