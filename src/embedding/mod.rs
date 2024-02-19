pub use routes::*;
use serde::{Deserialize, Serialize};
pub mod routes;
use fastembed::{EmbeddingModel, InitOptions, ModelInfo, TextEmbedding};
use schemars::JsonSchema;

const MODEL_NAME: &str = "fast-all-MiniLM-L6-v2";

#[derive(Clone, Deserialize, Serialize, JsonSchema, Debug)]
pub struct EmbeddingRequestObject {
    pub text_to_embed: String,
    pub id: i32,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponseObject {
    embedding: Vec<f32>,
    id: i32,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponse {
    number_of_documents: u32,
    total_time: tokio::time::Duration,
    time_per_document: tokio::time::Duration,
    embeddings: Vec<EmbeddingResponseObject>,
}

pub fn embed_documents(
    model: &TextEmbedding,
    request: Vec<EmbeddingRequestObject>,
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
    //calculate total time in milliseconds
    let response: EmbeddingResponse = EmbeddingResponse {
        embeddings: response_objects,
        number_of_documents: num_docs,
        total_time: end - start,
        time_per_document: (end - start) / num_docs,
    };
    response
}

pub fn get_model_info() -> Result<JSONModelInfo> {
    let models_info = TextEmbedding::list_supported_models();
    let models: Vec<ModelInfo> = models_info
        .into_iter()
        .filter(|s| s.model_code == *MODEL_NAME.to_string())
        .collect();
    if !models.is_empty() {
        let model = &models[0];
        let model: JSONModelInfo = JSONModelInfo {
            name: model.model_code.to_string(),
            dimension: model.dim as u32,
            description: model.description.clone(),
        };
        Ok(model)
    } else {
        Err(ModelNotFoundError)
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

#[derive(Serialize, Deserialize, JsonSchema)]
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

pub fn init_model() -> TextEmbedding {
    let model_name: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;
    new_model(model_name)
}

pub fn new_model(model_name: EmbeddingModel) -> TextEmbedding {
    let model: TextEmbedding = TextEmbedding::try_new(InitOptions {
        model_name,
        ..Default::default()
    })
    .expect("Can't load model");

    model
}
