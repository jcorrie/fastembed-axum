use std::sync::Arc;

use fastembed::TextEmbedding;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<TextEmbedding>,
}
