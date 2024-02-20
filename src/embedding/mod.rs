pub use routes::*;
use serde::{Deserialize, Serialize};
pub mod routes;
use fastembed::{EmbeddingModel, InitOptions, ModelInfo, TextEmbedding};
use schemars::JsonSchema;
use std::time::Duration;

#[derive(Clone, Deserialize, Serialize, JsonSchema, Debug)]
pub struct EmbeddingRequestUnit {
    pub id: i32,
    pub text_to_embed: String,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponseObject {
    id: i32,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponse {
    embeddings: Vec<EmbeddingResponseObject>,
    number_of_documents: u32,
    total_time_ms: u128,        //total time in milliseconds
    time_per_document_ms: u128, //time per document in milliseconds
}

pub fn embed_documents(
    model: &TextEmbedding,
    request: Vec<EmbeddingRequestUnit>,
) -> EmbeddingResponse {
    let start = tokio::time::Instant::now();
    let num_docs: u32 = request.len() as u32;
    // Extract texts and ids from the request objects
    let texts: Vec<_> = request.iter().map(|x| &x.text_to_embed).collect();
    let ids: Vec<_> = request.iter().map(|x| x.id).collect();
    let embeddings = model.embed(texts, None).unwrap();
    let response_objects: Vec<_> = embeddings
        .into_iter()
        .zip(ids)
        .map(|(embedding, id)| EmbeddingResponseObject { embedding, id })
        .collect();
    //get end time
    let end = tokio::time::Instant::now();
    let duration: Duration = end - start;

    //calculate total time in milliseconds
    let response: EmbeddingResponse = EmbeddingResponse {
        embeddings: response_objects,
        number_of_documents: num_docs,
        total_time_ms: duration.as_millis(),
        time_per_document_ms: (duration.as_millis()) / num_docs as u128,
    };
    response
}

pub fn get_current_model_info(current_model: &EmbeddingModel) -> Result<JSONModelInfo> {
    let models_info = TextEmbedding::list_supported_models();
    if let Some(model) = models_info.iter().find(|s| s.model == *current_model) {
        Ok(JSONModelInfo {
            name: model.model_code.to_string(),
            dimension: model.dim as u32,
            description: model.description.clone(),
        })
    } else {
        Err(ModelNotFoundError) // Assuming ModelError is an enum with ModelNotFoundError variant
    }
}

pub fn get_model_by_string(proposed_model: String) -> Result<EmbeddingModel> {
    let models_info = TextEmbedding::list_supported_models();
    let models: Vec<ModelInfo> = models_info
        .into_iter()
        .filter(|s| s.model_code == proposed_model)
        .collect();
    if !models.is_empty() {
        let model_info = &models[0].clone();
        let model: EmbeddingModel = model_info.model.clone();
        Ok(model)
    } else {
        Err(ModelNotFoundError)
    }
}

pub fn get_available_models() -> Vec<JSONModelInfo> {
    let models_info = TextEmbedding::list_supported_models();
    let json_models_info: Vec<JSONModelInfo> = models_info
        .into_iter()
        .map(|model_info| JSONModelInfo {
            name: model_info.model_code.to_string(),
            dimension: model_info.dim as u32,
            description: model_info.description.clone(),
        })
        .collect();
    json_models_info
}

#[derive(Serialize, Deserialize, Clone, JsonSchema)]
pub struct JSONModelInfo {
    pub name: String,
    pub dimension: u32,
    pub description: String,
}

type Result<T> = std::result::Result<T, ModelNotFoundError>;

#[derive(Debug, Clone)]
pub struct ModelNotFoundError;

impl std::fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "The model you have searched for has not been found")
    }
}

pub fn init_text_embedding() -> TextEmbedding {
    let model_name: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;
    new_text_embedding(&model_name)
}

pub fn new_text_embedding(model_name: &EmbeddingModel) -> TextEmbedding {
    TextEmbedding::try_new(InitOptions {
        cache_dir: Into::into("./.fastembed_cache"),
        model_name: model_name.clone(),
        ..Default::default()
    })
    .expect("Can't load model")
}
