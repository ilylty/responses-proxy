use crate::types::responses::Error;
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

/// POST /v1/responses/{response_id}/cancel — Cancel an in-progress response.
///
/// - `200`: Cancellation signal sent.
/// - `404`: No in-flight response found with that ID.
pub async fn cancel(
    State(state): State<crate::app::State>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    tracing::info!(response_id = %response_id, "Cancel request");

    let was_in_flight = state.store().cancel_in_flight(&response_id).await;
    if was_in_flight {
        tracing::info!(response_id = %response_id, "Cancel triggered");
        Ok(Json(
            serde_json::json!({"cancelled": true, "response_id": response_id}),
        ))
    } else {
        let err = Error::invalid_request_with_param(
            format!("No in-flight response found with ID '{response_id}'."),
            "response_id",
        );
        Err((StatusCode::NOT_FOUND, Json(err.to_http_json())))
    }
}
