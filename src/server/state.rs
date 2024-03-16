use std::sync::{Arc, Mutex};

use crate::embedding;
use fastembed::TextEmbedding;

#[derive(Clone)]
pub struct AppState {
    pub text_embedding: Arc<TextEmbedding>,
    pub model: Arc<Mutex<embedding::HFEmbeddingModelOrUserDefinedModel>>,
    pub model_info: embedding::JSONModelInfo,
}
