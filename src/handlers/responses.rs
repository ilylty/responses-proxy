use crate::convert::{
    chat_to_responses, items_to_chat_messages, output_to_input_items, responses_to_chat,
};
use crate::types::chat::{Completion as ChatCompletionResponse, Request as ChatRequest};
use crate::types::event::StreamEvent;
use crate::types::responses::{Error, Request as ResponsesRequest, ResponseStatus};
use crate::types::streaming::*;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event as SseEvent, KeepAlive},
    },
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// POST /v1/responses — handles both streaming and non-streaming responses.
pub async fn responses(
    State(state): State<crate::app::State>,
    Json(mut req): Json<ResponsesRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let provider = state
        .config()
        .models
        .get(&req.model)
        .cloned()
        .ok_or_else(|| {
            let err = Error::invalid_request(format!(
                "Unknown model: {}. Available: {:?}",
                req.model,
                state.config().models.keys()
            ));
            (StatusCode::BAD_REQUEST, Json(err.to_http_json()))
        })?;

    if !provider.rewrite.responses_in.is_empty() {
        let mut body = serde_json::to_value(&req).map_err(|e| {
            let err = Error::server_error(e.to_string());
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err.to_http_json()))
        })?;
        crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.responses_in).map_err(
            |message| {
                let err = Error::server_error(message);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(err.to_http_json()))
            },
        )?;
        req = serde_json::from_value(body).map_err(|e| {
            let err = Error::invalid_request(e.to_string());
            (StatusCode::BAD_REQUEST, Json(err.to_http_json()))
        })?;
    }

    let model = req.model.clone();
    let is_stream = req.stream;
    let provider_model = provider.model.clone();
    let endpoint = format!("{}/chat/completions", provider.base_url);

    // Build chat request (responses_to_chat fetches history + handles instructions)
    let (chat_req, full_input_messages) = {
        let mut cr = responses_to_chat(req.clone(), &state)
            .await
            .map_err(|unsupported| {
                let err = Error::invalid_request(format!(
                    "Unsupported features: {}",
                    unsupported.join(", ")
                ));
                (StatusCode::BAD_REQUEST, Json(err.to_http_json()))
            })?;
        cr.model = provider_model.clone();
        let input_msgs = cr.messages.clone();
        (cr, input_msgs)
    };
    tracing::info!(
        model = %model,
        upstream = %provider_model,
        messages = chat_req.messages.len(),
        stream = is_stream,
        endpoint = %endpoint,
        "Forwarding request"
    );

    // Handle background mode: return queued status immediately, process in background
    if req.background && !is_stream {
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
        let queued_resp = crate::types::responses::Response {
            id: response_id.clone(),
            status: ResponseStatus::Queued,
            model: model.clone(),
            output: vec![],
            ..Default::default()
        };

        if req.store {
            let mut qr = queued_resp.clone();
            qr.previous_response_id = req.previous_response_id.clone();

            let messages = items_to_chat_messages(&req.input, &state);
            state.store().put(response_id.clone(), messages).await;
        }

        // Capture full input messages for background task storage
        let bg_full_input = full_input_messages.clone();

        // Spawn background processing
        let bg_state = state.clone();
        let bg_provider = provider.clone();
        let bg_model = model.clone();
        let bg_req = req.clone();
        let bg_rid = response_id.clone();

        tokio::spawn(async move {
            let result =
                execute_upstream_request(&bg_state, &bg_provider, chat_req, bg_model, &bg_req)
                    .await;

            match result {
                Ok(mut resp) => {
                    if let Some(ref meta) = bg_req.metadata {
                        resp.metadata = Some(meta.clone());
                    }
                    resp.previous_response_id = bg_req.previous_response_id.clone();

                    let mut bg_messages = bg_full_input;
                    let output_inputs = output_to_input_items(&resp.output);
                    bg_messages.extend(items_to_chat_messages(&output_inputs, &bg_state));
                    bg_state.store().put(bg_rid, bg_messages).await;
                }
                Err(err) => {
                    let rid = bg_rid.clone();
                    bg_state.store().put(bg_rid, bg_full_input).await;

                    tracing::error!(
                        response_id = %rid,
                        error = %err,
                        "Background response processing failed"
                    );
                }
            }
        });

        return Ok((StatusCode::ACCEPTED, Json(queued_resp)).into_response());
    }

    if is_stream {
        handle_streaming(&state, &provider, chat_req, model, req, full_input_messages)
            .await
            .map(|s| s.into_response())
    } else {
        handle_non_streaming(&state, &provider, chat_req, model, req, full_input_messages)
            .await
            .map(|j| j.into_response())
    }
}

// ── Shared upstream call ──────────────────────────────────────────────────

fn send_chat_request(
    request: reqwest::RequestBuilder,
    chat_req: &ChatRequest,
    provider: &crate::config::ResolvedProvider,
) -> Result<reqwest::RequestBuilder, String> {
    if provider.rewrite.chat_out.is_empty() {
        return Ok(request.json(chat_req));
    }

    let mut body = serde_json::to_value(chat_req).map_err(|e| e.to_string())?;
    crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.chat_out)?;
    Ok(request.json(&body))
}

/// Call upstream Chat API, parse the response, convert to Responses format,
/// and apply include-based trimming. Does NOT handle persistence or metadata.
async fn execute_upstream_request(
    state: &crate::app::State,
    provider: &crate::config::ResolvedProvider,
    chat_req: ChatRequest,
    model: String,
    original_req: &ResponsesRequest,
) -> Result<crate::types::responses::Response, String> {
    let url = format!("{}/chat/completions", provider.base_url);

    let request = state
        .http_client()
        .post(url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json");
    let response = send_chat_request(request, &chat_req, provider)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = response.status();
    let body = response.text().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(format!(
            "Upstream returned {status}: {}",
            if body.len() > 200 {
                &body[..200]
            } else {
                &body
            }
        ));
    }

    let chat_resp: ChatCompletionResponse = if provider.rewrite.chat_in.is_empty() {
        serde_json::from_str(&body).map_err(|e| e.to_string())?
    } else {
        let mut chat_response_body: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| e.to_string())?;
        crate::rewrite::apply_rewrite(&mut chat_response_body, &provider.rewrite.chat_in)?;
        serde_json::from_value(chat_response_body).map_err(|e| e.to_string())?
    };

    let mut resp = chat_to_responses(chat_resp, model, state.compact_key());

    apply_include_filter(&mut resp, &original_req.include);

    Ok(resp)
}

// ── Non-streaming handler ────────────────────────────────────────────────

async fn handle_non_streaming(
    state: &crate::app::State,
    provider: &crate::config::ResolvedProvider,
    chat_req: ChatRequest,
    model: String,
    original_req: ResponsesRequest,
    full_input_messages: Vec<crate::types::chat::MessageRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let mut resp = execute_upstream_request(state, provider, chat_req, model, &original_req)
        .await
        .map_err(|msg| {
            let err = Error::server_error(msg);
            (StatusCode::BAD_GATEWAY, Json(err.to_http_json()))
        })?;

    // Persist to store if store=true (default)
    if original_req.store {
        let mut stored_resp = resp.clone();
        if let Some(ref meta) = original_req.metadata {
            stored_resp.metadata = Some(meta.clone());
        }
        stored_resp.previous_response_id = original_req.previous_response_id.clone();
        let mut store_messages = full_input_messages;
        let output_inputs = output_to_input_items(&resp.output);
        store_messages.extend(items_to_chat_messages(&output_inputs, state));
        state.store().put(resp.id.clone(), store_messages).await;
    }

    // Echo back request metadata and other params
    if let Some(ref meta) = original_req.metadata {
        resp.metadata = Some(meta.clone());
    }
    resp.parallel_tool_calls = original_req.parallel_tool_calls;

    if provider.rewrite.responses_out.is_empty() {
        return Ok(Json(resp).into_response());
    }

    let mut response_body = serde_json::to_value(&resp).map_err(|e| {
        let err = Error::server_error(e.to_string());
        (StatusCode::INTERNAL_SERVER_ERROR, Json(err.to_http_json()))
    })?;
    crate::rewrite::apply_rewrite(&mut response_body, &provider.rewrite.responses_out).map_err(
        |message| {
            let err = Error::server_error(message);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err.to_http_json()))
        },
    )?;

    Ok(Json(response_body).into_response())
}

// ── Streaming (SSE) handler ──────────────────────────────────────────────

async fn handle_streaming(
    state: &crate::app::State,
    provider: &crate::config::ResolvedProvider,
    chat_req: ChatRequest,
    model: String,
    original_req: ResponsesRequest,
    full_input_messages: Vec<crate::types::chat::MessageRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let url = format!("{}/chat/completions", provider.base_url);

    let request = state
        .http_client()
        .post(url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json");
    let response = send_chat_request(request, &chat_req, provider)
        .map_err(|message| {
            let err = Error::server_error(message);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err.to_http_json()))
        })?
        .send()
        .await
        .map_err(|e| {
            let err = Error::server_error(e.to_string());
            (StatusCode::BAD_GATEWAY, Json(err.to_http_json()))
        })?;

    if !response.status().is_success() {
        let s = response.status();
        let b = response.text().await.unwrap_or_default();
        let truncated = if b.len() > 200 { &b[..200] } else { &b };
        let err = Error::server_error(format!("Upstream returned {}: {}", s.as_u16(), truncated));
        return Err((StatusCode::BAD_GATEWAY, Json(err.to_http_json())));
    }

    // SSE → mpsc channel so we can stream to the client
    let (tx, rx) = mpsc::channel::<Result<SseEvent, std::convert::Infallible>>(64);
    let rid = format!("resp_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let mid = format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));

    let mut bytes = response.bytes_stream();
    let store_events = original_req.store;
    let store = state.store().clone();
    let chat_in_rewrite = provider.rewrite.chat_in.clone();
    let responses_out_rewrite = provider.rewrite.responses_out.clone();
    let bg_state = state.clone();

    // Register cancellation token so POST /v1/responses/{id}/cancel can stop this stream
    let cancel_rx = store.register_cancel_token(&rid).await;

    tokio::spawn(async move {
        let mut buf = String::new();
        let mut ss = StreamState::new(rid.clone(), mid.clone(), model.clone());
        let seq: u64 = 0;
        let mut collected_events: Vec<StreamEvent> = Vec::new();
        let mut cancel_rx = cancel_rx;
        ss.created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        ss.has_started = true;
        ss.compact_key = bg_state.compact_key().copied();

        let start_response =
            build_stream_lifecycle_response(&rid, &model, ss.created, ResponseStatus::InProgress);
        let start_events = [
            StreamEvent::Created(crate::types::event::Created {
                response: start_response.clone(),
                sequence_number: 0,
            }),
            StreamEvent::InProgress(crate::types::event::InProgress {
                response: start_response,
                sequence_number: 1,
            }),
        ];
        for event in start_events {
            let prepared = match prepare_stream_event(event, &responses_out_rewrite) {
                Ok(prepared) => prepared,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to prepare SSE lifecycle event");
                    store.unregister_cancel_token(&rid).await;
                    return;
                }
            };
            if store_events {
                collected_events.push(prepared.event);
            }
            let sse_event = match SseEvent::default()
                .event(prepared.event_type)
                .json_data(prepared.body)
            {
                Ok(event) => event,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to build SSE lifecycle event");
                    store.unregister_cancel_token(&rid).await;
                    return;
                }
            };
            if tx.send(Ok(sse_event)).await.is_err() {
                store.unregister_cancel_token(&rid).await;
                return;
            }
        }

        loop {
            let chunk = tokio::select! {
                _ = cancel_rx.changed() => {
                    tracing::info!(response_id = %rid, "SSE stream cancelled");
                    break;
                }
                chunk = bytes.next() => chunk,
            };

            match chunk {
                Some(Ok(b)) => {
                    buf.push_str(&String::from_utf8_lossy(&b));
                }
                _ => break,
            }

            // Parse SSE events (delimited by \n\n)
            while let Some(pos) = buf.find("\n\n") {
                let event = buf[..pos].trim().to_string();
                buf = buf[pos + 2..].to_string();

                if let Some(data) = event
                    .lines()
                    .find(|l| l.starts_with("data:"))
                    .and_then(|l| l.strip_prefix("data:").map(|s| s.trim()))
                {
                    tracing::trace!(%data, "Chat API delta");

                    match process_upstream_stream_data(
                        &mut ss,
                        data,
                        &chat_in_rewrite,
                        &responses_out_rewrite,
                    ) {
                        Ok(events) => {
                            for prepared in events {
                                if store_events {
                                    collected_events.push(prepared.event);
                                }

                                let sse_event = match SseEvent::default()
                                    .event(prepared.event_type)
                                    .json_data(prepared.body)
                                {
                                    Ok(e) => e,
                                    Err(e) => {
                                        tracing::error!(error = %e, "Failed to build SSE event");
                                        continue;
                                    }
                                };
                                if tx.send(Ok(sse_event)).await.is_err() {
                                    // Client disconnected — persist partial events and exit
                                    if store_events && !collected_events.is_empty() {
                                        let final_resp = build_response_from_state(&ss);
                                        let mut msgs = full_input_messages.clone();
                                        let out = output_to_input_items(&final_resp.output);
                                        msgs.extend(items_to_chat_messages(&out, &bg_state));
                                        store.put(rid.clone(), msgs).await;
                                    }
                                    store.unregister_cancel_token(&rid).await;
                                    return;
                                }
                            }
                        }
                        Err(message) => {
                            let error_event = StreamEvent::Error(crate::types::event::Error {
                                code: Some(Error::CODE_SERVER_ERROR.into()),
                                message,
                                param: None,
                                sequence_number: seq as i64,
                            });
                            let prepared = match prepare_stream_event(
                                error_event,
                                &responses_out_rewrite,
                            ) {
                                Ok(e) => e,
                                Err(e) => {
                                    tracing::error!(error = %e, "Failed to prepare SSE error event");
                                    store.unregister_cancel_token(&rid).await;
                                    return;
                                }
                            };
                            let sse_event = match SseEvent::default()
                                .event(prepared.event_type)
                                .json_data(prepared.body)
                            {
                                Ok(e) => e,
                                Err(e) => {
                                    tracing::error!(error = %e, "Failed to build SSE error event");
                                    store.unregister_cancel_token(&rid).await;
                                    return;
                                }
                            };
                            let _ = tx.send(Ok(sse_event)).await;
                            store.unregister_cancel_token(&rid).await;
                            return;
                        }
                    }
                }
            }
        }

        // Persist completed response to store
        if store_events {
            let final_resp = build_response_from_state(&ss);
            let output_count = final_resp.output.len();
            let has_reasoning_output = final_resp
                .output
                .iter()
                .any(|o| matches!(o, crate::types::item::OutputItem::Reasoning(_)));
            let mut msgs = full_input_messages;
            let out = output_to_input_items(&final_resp.output);
            let chat_msgs = items_to_chat_messages(&out, &bg_state);
            let has_reasoning_in_stored = chat_msgs.iter().any(|m| match m {
                crate::types::chat::MessageRequest::Assistant(a) => a.reasoning_content.is_some(),
                _ => false,
            });
            tracing::info!(
                output_count,
                has_reasoning_output,
                has_reasoning_in_stored,
                reasoning_open = %ss.reasoning_content,
                completed = ss.completed_items.len(),
                "SSE: persisting response"
            );
            msgs.extend(chat_msgs);
            store.put(rid.clone(), msgs).await;
        }
        store.unregister_cancel_token(&rid).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()))
}

fn build_stream_lifecycle_response(
    response_id: &str,
    model: &str,
    created_at: i64,
    status: ResponseStatus,
) -> crate::types::responses::Response {
    crate::types::responses::Response {
        id: response_id.to_string(),
        model: model.to_string(),
        status,
        created_at,
        output: vec![],
        ..Default::default()
    }
}

/// Build a Response from the streaming state for store persistence.
pub(crate) fn build_response_from_state(ss: &StreamState) -> crate::types::responses::Response {
    // Start with items that were closed mid-stream (e.g. reasoning→text transitions)
    let mut output_items: Vec<crate::types::item::OutputItem> = ss.completed_items.clone();

    // Reasoning item — still open if content remains
    if !ss.reasoning_content.is_empty() {
        output_items.push(crate::types::item::OutputItem::Reasoning(
            crate::types::streaming::build_reasoning_item(
                ss.reasoning_id.clone(),
                &ss.reasoning_content,
                ss.compact_key.as_ref(),
            ),
        ));
    }

    // Function call items — open if id is set (not cleared by close)
    for tc in &ss.tool_calls {
        if tc.id.is_empty() {
            continue;
        }
        use crate::types::item::{FunctionCall, OutputItem};
        output_items.push(OutputItem::FunctionCall(FunctionCall {
            call_id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            id: Some(tc.fc_id.clone()),
            namespace: None,
            status: Some("completed".into()),
        }));
    }

    // Message item — still open if content remains
    if !ss.accumulated_text.is_empty() {
        use crate::types::item::{OutputContentBlock, OutputItem, OutputMessage};
        let content = if ss.has_refusal {
            vec![OutputContentBlock::Refusal {
                refusal: ss.accumulated_text.clone(),
            }]
        } else {
            vec![OutputContentBlock::Text {
                text: ss.accumulated_text.clone(),
                annotations: vec![],
                logprobs: None,
            }]
        };
        output_items.push(OutputItem::Message(OutputMessage {
            id: ss.msg_id.clone(),
            role: "assistant".into(),
            status: "completed".into(),
            content,
            phase: None,
        }));
    }

    let status = if ss.has_refusal {
        ResponseStatus::Incomplete
    } else {
        ResponseStatus::Completed
    };

    let usage = ss
        .usage
        .as_ref()
        .map(|u| crate::types::responses::Usage::from(u.clone()));

    crate::types::responses::Response {
        id: ss.response_id.clone(),
        model: ss.model.clone(),
        status,
        created_at: ss.created,
        output: output_items,
        usage,
        ..Default::default()
    }
}

/// Post-process a Response to trim fields based on the `include` values.
///
/// If `include` is None or empty, no trimming occurs.
/// If `include` does NOT contain `message.output_text.logprobs`, logprobs are
/// stripped from all output text blocks.
fn apply_include_filter(
    resp: &mut crate::types::responses::Response,
    include: &Option<Vec<crate::types::responses::Include>>,
) {
    let Some(includes) = include else { return };
    if includes.is_empty() {
        return;
    }

    let want_logprobs = includes
        .iter()
        .any(|inc| inc.as_ref() == crate::types::responses::Include::MESSAGE_OUTPUT_TEXT_LOGPROBS);

    if !want_logprobs {
        for item in &mut resp.output {
            if let crate::types::item::OutputItem::Message(msg) = item {
                for block in &mut msg.content {
                    if let crate::types::item::OutputContentBlock::Text { logprobs, .. } = block {
                        *logprobs = None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RewriteConfig, RewriteStep};
    use crate::types::chat::{CompletionTokensDetails, PromptTokensDetails, Usage};
    use crate::types::item::{OutputContentBlock, OutputItem, OutputMessage};
    use crate::types::responses::{Include, Response, ResponseStatus};

    // ── apply_include_filter ────────────────────────────────────────────────

    fn make_text_output_resp() -> Response {
        Response {
            id: "resp_1".into(),
            model: "gpt-5".into(),
            status: ResponseStatus::Completed,
            output: vec![OutputItem::Message(OutputMessage {
                id: "msg_1".into(),
                role: "assistant".into(),
                status: "completed".into(),
                content: vec![OutputContentBlock::Text {
                    text: "hello".into(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                }],
                phase: None,
            })],
            ..Default::default()
        }
    }

    #[test]
    fn include_filter_none_does_nothing() {
        let mut resp = make_text_output_resp();
        apply_include_filter(&mut resp, &None);
        // logprobs should be preserved
        if let OutputItem::Message(msg) = &resp.output[0]
            && let OutputContentBlock::Text { logprobs, .. } = &msg.content[0]
        {
            assert!(
                logprobs.is_some(),
                "logprobs should be preserved when include is None"
            );
        }
    }

    #[test]
    fn include_filter_empty_does_nothing() {
        let mut resp = make_text_output_resp();
        apply_include_filter(&mut resp, &Some(vec![]));
        if let OutputItem::Message(msg) = &resp.output[0]
            && let OutputContentBlock::Text { logprobs, .. } = &msg.content[0]
        {
            assert!(
                logprobs.is_some(),
                "logprobs should be preserved when include is empty"
            );
        }
    }

    #[test]
    fn include_filter_requested_logprobs_preserves_them() {
        let mut resp = make_text_output_resp();
        apply_include_filter(
            &mut resp,
            &Some(vec![Include(
                Include::MESSAGE_OUTPUT_TEXT_LOGPROBS.to_string(),
            )]),
        );
        if let OutputItem::Message(msg) = &resp.output[0]
            && let OutputContentBlock::Text { logprobs, .. } = &msg.content[0]
        {
            assert!(
                logprobs.is_some(),
                "logprobs should be preserved when requested"
            );
        }
    }

    #[test]
    fn include_filter_without_logprobs_strips_them() {
        let mut resp = make_text_output_resp();
        apply_include_filter(
            &mut resp,
            &Some(vec![Include("web_search_call.results".into())]),
        );
        if let OutputItem::Message(msg) = &resp.output[0]
            && let OutputContentBlock::Text { logprobs, .. } = &msg.content[0]
        {
            assert!(
                logprobs.is_none(),
                "logprobs should be stripped when not requested"
            );
        }
    }

    #[test]
    fn stream_data_applies_chat_in_before_chunk_parse() {
        let mut ss = StreamState::new("resp_s".into(), "msg_s".into(), "gpt-5".into());
        let chat_in = RewriteConfig {
            steps: vec![RewriteStep::Reset(vec![(
                "created".into(),
                serde_json::json!(42),
            )])],
        };
        let events = process_upstream_stream_data(
            &mut ss,
            r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#,
            &chat_in,
            &RewriteConfig::default(),
        )
        .unwrap();

        assert_eq!(ss.created, 42);
        assert_eq!(events.len(), 5);
        assert_eq!(events[0].body["sequence_number"], 3);
        assert_eq!(events[4].body["type"], "response.output_text.delta");
    }

    #[test]
    fn stream_data_applies_response_out_to_event_body_only() {
        let mut ss = StreamState::new("resp_s".into(), "msg_s".into(), "gpt-5".into());
        let responses_out = RewriteConfig {
            steps: vec![RewriteStep::Remove(vec!["response.output".into()])],
        };
        let events = process_upstream_stream_data(
            &mut ss,
            r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"t","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#,
            &RewriteConfig::default(),
            &responses_out,
        )
        .unwrap();

        assert!(events[0].body["response"].get("output").is_none());
        if let StreamEvent::Created(created) = &events[0].event {
            assert!(created.response.output.is_empty());
        } else {
            panic!("expected response.created event");
        }
    }

    // ── build_response_from_state ───────────────────────────────────────────

    fn basic_stream_state() -> StreamState {
        let mut ss = StreamState::new("resp_s".into(), "msg_s".into(), "gpt-5".into());
        ss.accumulated_text.push_str("test");
        ss.accumulated_text = "hello world".into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        ss.created = now;
        ss
    }

    #[test]
    fn build_response_basic_text() {
        let ss = basic_stream_state();
        let resp = build_response_from_state(&ss);
        assert_eq!(resp.id, "resp_s");
        assert_eq!(resp.model, "gpt-5");
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 1);
        if let OutputItem::Message(msg) = &resp.output[0] {
            assert_eq!(msg.id, "msg_s");
            assert_eq!(msg.role, "assistant");
            assert_eq!(msg.status, "completed");
            assert_eq!(msg.content.len(), 1);
            if let OutputContentBlock::Text { text, .. } = &msg.content[0] {
                assert_eq!(text, "hello world");
            }
        }
    }

    #[test]
    fn build_response_with_reasoning() {
        use crate::types::streaming::StreamState as Ss;
        let mut ss = Ss::new("resp_r".into(), "msg_r".into(), "gpt-5".into());
        ss.reasoning_content.push_str("test");
        ss.reasoning_id = "rsn_1".into();
        ss.reasoning_content = "I need to think...".into();
        ss.accumulated_text.push_str("test");
        ss.accumulated_text = "answer".into();

        let resp = build_response_from_state(&ss);
        assert_eq!(resp.output.len(), 2);
        // First output should be reasoning
        if let OutputItem::Reasoning(r) = &resp.output[0] {
            assert_eq!(r.id, "rsn_1");
            assert_eq!(r.status.as_deref(), Some("completed"));
            assert!(r.content.is_some());
        } else {
            panic!("expected reasoning item, got {:?}", resp.output[0]);
        }
        // Second should be message
        if let OutputItem::Message(msg) = &resp.output[1]
            && let OutputContentBlock::Text { text, .. } = &msg.content[0]
        {
            assert_eq!(text, "answer");
        }
    }

    #[test]
    fn build_response_with_refusal() {
        let mut ss = StreamState::new("resp_ref".into(), "msg_ref".into(), "gpt-5".into());
        ss.accumulated_text.push_str("test");
        ss.has_refusal = true;
        ss.accumulated_text = "I cannot answer that.".into();

        let resp = build_response_from_state(&ss);
        assert_eq!(resp.status, ResponseStatus::Incomplete);
        if let OutputItem::Message(msg) = &resp.output[0]
            && let OutputContentBlock::Refusal { refusal } = &msg.content[0]
        {
            assert_eq!(refusal, "I cannot answer that.");
        }
    }

    #[test]
    fn build_response_with_function_calls() {
        let mut ss = StreamState::new("resp_fc".into(), "msg_fc".into(), "gpt-5".into());
        ss.tool_calls = vec![
            crate::types::streaming::ToolCallAccumulator {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"NYC"}"#.into(),
                fc_id: "fc_1".into(),
                index: 0,
                output_index: 0,
            },
            crate::types::streaming::ToolCallAccumulator {
                id: String::new(), // skipped — empty id
                name: "ignored".into(),
                arguments: String::new(),
                fc_id: String::new(),
                index: 1,
                output_index: 1,
            },
        ];

        let resp = build_response_from_state(&ss);
        // Only the function call (no text content was added)
        assert_eq!(resp.output.len(), 1);
        if let OutputItem::FunctionCall(fc) = &resp.output[0] {
            assert_eq!(fc.name, "get_weather");
            assert_eq!(fc.arguments, r#"{"city":"NYC"}"#);
        } else {
            panic!("expected function call");
        }
        // Empty-id tool call should be skipped
        let fc_count = resp
            .output
            .iter()
            .filter(|o| matches!(o, OutputItem::FunctionCall(_)))
            .count();
        assert_eq!(fc_count, 1);
    }

    #[test]
    fn build_response_with_usage() {
        let mut ss = basic_stream_state();
        ss.usage = Some(Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 20,
                audio_tokens: 0,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: 30,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            }),
        });

        let resp = build_response_from_state(&ss);
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.output_tokens, 50);
        assert_eq!(u.total_tokens, 150);
        assert_eq!(u.input_tokens_details.cached_tokens, 20);
        assert_eq!(u.output_tokens_details.reasoning_tokens, 30);
    }

    #[test]
    fn build_response_empty_state() {
        let ss = StreamState::new("resp_empty".into(), "msg_e".into(), "gpt-5".into());
        let resp = build_response_from_state(&ss);
        assert_eq!(resp.id, "resp_empty");
        assert_eq!(resp.output.len(), 0);
        assert!(resp.usage.is_none());
    }
}
