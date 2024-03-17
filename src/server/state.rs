use crate::embedding::{self, HFEmbeddingModelOrUserDefinedModel};
use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, TextEmbedding};

#[derive(Clone)]
pub struct AppState {
    pub text_embedding: Arc<TextEmbedding>,
    pub model: Arc<Mutex<embedding::HFEmbeddingModelOrUserDefinedModel>>,
    pub model_info: embedding::JSONModelInfo,
}

pub async fn get_app_state(model_source: embedding::ModelSource) -> AppState {
    let state: AppState = match model_source {
        embedding::ModelSource::HuggingFace => {
            let hf_embedding_model = EmbeddingModel::BGEBaseENV15;
            let embedding_model = embedding::HFEmbeddingModelOrUserDefinedModel::HuggingFace(
                hf_embedding_model.clone(),
            );
            let model_info: embedding::JSONModelInfo =
                embedding::get_current_model_info(&embedding_model).expect("Can't load model");
            let text_embedding: TextEmbedding = embedding::new_text_embedding(&hf_embedding_model);
            AppState {
                text_embedding: Arc::new(text_embedding),
                model: Arc::new(Mutex::new(embedding_model)),
                model_info,
            }
        }
        embedding::ModelSource::Local(model) => {
            let model_info: embedding::JSONModelInfo = embedding::get_current_model_info(
                &HFEmbeddingModelOrUserDefinedModel::UserDefined(model.clone()),
            )
            .expect("Can't load model");
            let text_embedding: TextEmbedding =
                embedding::new_text_embedding_user_defined(model.clone());
            AppState {
                text_embedding: Arc::new(text_embedding),
                model: Arc::new(Mutex::new(HFEmbeddingModelOrUserDefinedModel::UserDefined(
                    model,
                ))),
                model_info,
            }
        }
    };
    state
}
