use std::borrow::Borrow;

use aide::{
    axum::{
        routing::{get_with, post_with},
        ApiRouter,
    },
    transform::TransformOperation,
};

use axum::{extract::State, http::StatusCode};

use fastembed::UserDefinedEmbeddingModel;
use schemars::JsonSchema;

use crate::{server::extractors::Json, server::state::AppState};

use super::{
    embed_documents, get_available_models, get_current_model_info, get_model_by_string,
    EmbeddingRequestUnit, EmbeddingResponse, HFEmbeddingModelOrUserDefinedModel, JSONModelInfo,
    ModelNotFoundError,
};
use axum_macros::debug_handler;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema, Debug)]
pub struct EmbeddingRequest {
    data: Vec<EmbeddingRequestUnit>,
}

pub fn embed_routes(state: AppState) -> ApiRouter {
    ApiRouter::new()
        .api_route("/generate", post_with(embed, all_docs))
        .api_route("/model-info", get_with(model_info, all_docs))
        .api_route("/set-model-name", post_with(url_set_model_name, all_docs))
        .api_route("/available-models", get_with(available_models, all_docs))
        .with_state(state)
}

fn all_docs(op: TransformOperation) -> TransformOperation {
    op.description("Documentation")
        .response::<201, Json<EmbeddingRequest>>()
}

#[debug_handler]
pub async fn embed(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingRequest>,
) -> (StatusCode, Json<EmbeddingResponse>) {
    let embeddings = embed_documents(&state.text_embedding, payload.data);
    (StatusCode::ACCEPTED, Json(embeddings))
}

pub async fn model_info(State(state): State<AppState>) -> (StatusCode, Json<JSONModelInfo>) {
    let model_guard = state.model.lock().unwrap();
    let model = &*model_guard;
    let model_info: Result<JSONModelInfo, ModelNotFoundError> = match model {
        HFEmbeddingModelOrUserDefinedModel::HuggingFace(model) => {
            get_current_model_info(&model)
        }
        HFEmbeddingModelOrUserDefinedModel::UserDefined(model) => {
            Ok(JSONModelInfo {
                name: model.model_code.clone(),
                dimension: model.dim as u32,
                description: model.description.clone(),
            })
        }
    };
    match model_info {
        Ok(model) => (StatusCode::OK, Json(model)),
        Err(ModelNotFoundError) => (
            StatusCode::NOT_FOUND,
            Json(JSONModelInfo {
                name: "".to_string(),
                dimension: 0,
                description: "".to_string(),
            }),
        ),
    }
}

#[debug_handler]
pub async fn url_set_model_name(
    State(state): State<AppState>,
    Json(payload): Json<SetModelName>,
) -> StatusCode {
    let model_result: Result<fastembed::EmbeddingModel, ModelNotFoundError> =
        get_model_by_string(payload.model);
    // If the model is not found, return a 404, else set the model

    match model_result {
        Ok(model) => {
            // If the model is found, update the state
            let mut model_guard = state.model.lock().unwrap();
            *model_guard = HFEmbeddingModelOrUserDefinedModel::HuggingFace(model);
            StatusCode::CREATED
        }
        Err(ModelNotFoundError) => StatusCode::NOT_FOUND,
    }
}

#[debug_handler]
pub async fn available_models() -> (StatusCode, Json<Vec<JSONModelInfo>>) {
    let models: Vec<JSONModelInfo> = get_available_models();
    (StatusCode::OK, Json(models))
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct SetModelName {
    model: String,
}

pub async fn hello_world() -> (StatusCode, Json<String>) {
    (StatusCode::OK, Json("Hello!".to_string()))
}
