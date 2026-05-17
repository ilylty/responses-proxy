//! Shared upstream Chat Completions API request builder.

use crate::config::ResolvedProvider;
use crate::types::chat;

/// Build a POST request to the upstream Chat Completions endpoint.
pub fn build_chat_request(
    client: &reqwest::Client,
    provider: &ResolvedProvider,
    body: &serde_json::Value,
) -> reqwest::RequestBuilder {
    let url = format!("{}/chat/completions", provider.base_url);
    client
        .post(&url)
        .timeout(provider.timeout)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header("Content-Type", "application/json")
        .json(body)
}

/// Build a POST request from a typed ChatRequest, applying `chat_out` rewrite if configured.
pub fn build_typed_chat_request(
    client: &reqwest::Client,
    provider: &ResolvedProvider,
    chat_req: &chat::Request,
) -> Result<reqwest::RequestBuilder, String> {
    if provider.rewrite.chat_out.is_empty() {
        let body = serde_json::to_value(chat_req).map_err(|e| e.to_string())?;
        return Ok(build_chat_request(client, provider, &body));
    }

    let mut body = serde_json::to_value(chat_req).map_err(|e| e.to_string())?;
    crate::rewrite::apply_rewrite(&mut body, &provider.rewrite.chat_out)?;
    Ok(build_chat_request(client, provider, &body))
}
