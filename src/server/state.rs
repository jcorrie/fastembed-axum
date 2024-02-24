use std::sync::{Arc, Mutex};

use crate::embedding;
use fastembed::{EmbeddingModel, TextEmbedding};

#[derive(Clone)]
pub struct AppState {
    pub text_embedding: Arc<TextEmbedding>,
    pub model: Arc<Mutex<EmbeddingModel>>,
    pub model_info: embedding::JSONModelInfo,
}
