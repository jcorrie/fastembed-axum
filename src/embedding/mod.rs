pub mod routes;

pub use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, ModelInfo, TextEmbedding,
    UserDefinedEmbeddingModel,
};

use reqwest;

pub use routes::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{io::Read, path::PathBuf, time::Duration};

pub enum HFEmbeddingModelOrUserDefinedModel {
    HuggingFace(EmbeddingModel),
    UserDefined(Box<UserDefinedEmbeddingModel>),
}

pub enum ModelSource {
    HuggingFace,
    Local(Box<UserDefinedEmbeddingModel>),
}
#[derive(Clone, Deserialize, Serialize, JsonSchema, Debug)]
pub struct EmbeddingRequestUnit {
    pub id: i32,
    pub text_to_embed: String,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponseObject {
    id: i32,
    embeddings: Vec<Vec<f32>>, //Vec of vecs, so we can store multiple embeddings for each document
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
pub struct EmbeddingResponse {
    number_of_documents: u32,
    total_time_ms: u128,        //total time in milliseconds
    time_per_document_ms: u128, //time per document in milliseconds
    embeddings: Vec<EmbeddingResponseObject>,
}

#[derive(Serialize, Deserialize, Debug, JsonSchema)]
struct EmbeddingTracker {
    id: i32,
    num_docs: u32,
    text: Vec<String>,
}

pub fn embed_documents(
    model: &TextEmbedding,
    request: Vec<EmbeddingRequestUnit>,
) -> EmbeddingResponse {
    let start = tokio::time::Instant::now();
    let num_docs: u32 = request.len() as u32;
    let ids: Vec<_> = request.iter().map(|x| x.id).collect();
    let texts: Vec<String> = request.iter().map(|x| x.text_to_embed.clone()).collect();
    let mut embedding_trackers: Vec<EmbeddingTracker> = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let chunked_texts = chunk_with_overlap(text, 10, 3);
        let tracker = EmbeddingTracker {
            id: ids[i],
            num_docs: chunked_texts.len() as u32,
            text: chunked_texts,
        };
        embedding_trackers.push(tracker);
    }
    println!("{:?}", embedding_trackers);
    let flattened_chunked_texts: Vec<String> = embedding_trackers
        .iter()
        .flat_map(|tracker| tracker.text.clone())
        .collect();

    let embeddings_vec: Vec<Vec<f32>> = model.embed(flattened_chunked_texts, None).unwrap();
    // Rebuild the embeddings into the original structure
    let mut embeddings: Vec<EmbeddingResponseObject> = Vec::new();
    for tracker in embedding_trackers {
        let mut embeddings_for_doc: Vec<Vec<f32>> = Vec::new();
        for chunk in tracker.text {
            let chunk_embeddings: Vec<f32> = embeddings_vec
                .iter()
                .take(chunk.len())
                .cloned()
                .collect::<Vec<Vec<f32>>>()
                .concat();
            embeddings_for_doc.push(chunk_embeddings);
        }
        let embeddings_object = EmbeddingResponseObject {
            id: tracker.id,
            embeddings: embeddings_for_doc,
        };
        embeddings.push(embeddings_object);
    }

    //get end time
    let end = tokio::time::Instant::now();
    let duration: Duration = end - start;

    //calculate total time in milliseconds
    let response: EmbeddingResponse = EmbeddingResponse {
        embeddings,
        number_of_documents: num_docs,
        total_time_ms: duration.as_millis(),
        time_per_document_ms: (duration.as_millis()) / num_docs as u128,
    };
    response
}

fn chunk_with_overlap(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = std::cmp::min(i + chunk_size, chars.len());
        let chunk: String = chars[i..end].iter().collect();
        chunks.push(chunk);
        i = if end == chars.len() {
            end
        } else {
            end - overlap
        };
    }
    chunks
}

pub fn get_current_model_info(
    current_model: &HFEmbeddingModelOrUserDefinedModel,
) -> Result<JSONModelInfo, ModelNotFoundError> {
    match current_model {
        HFEmbeddingModelOrUserDefinedModel::HuggingFace(model) => {
            let model_info: ModelInfo =
                TextEmbedding::get_model_info(model).expect("Model not found");
            Ok(JSONModelInfo {
                name: model_info.model_code.to_string(),
                dimension: model_info.dim as u32,
                description: model_info.description.clone(),
            })
        }
        HFEmbeddingModelOrUserDefinedModel::UserDefined(model) => Ok(JSONModelInfo {
            name: model.model_code.to_string(),
            dimension: model.dim as u32,
            description: model.description.clone(),
        }),
    }
}

pub fn get_model_by_string(proposed_model: String) -> Result<EmbeddingModel, ModelNotFoundError> {
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

pub fn new_text_embedding_user_defined(model: UserDefinedEmbeddingModel) -> TextEmbedding {
    TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined {
            ..Default::default()
        },
    )
    .expect("Can't load model")
}

#[derive(Serialize, Deserialize, JsonSchema, Debug)]
pub enum LocalOrRemoteFile {
    Local(PathBuf),
    Remote(String),
}

impl LocalOrRemoteFile {
    pub fn read_local_or_remote_file_to_bytes(self) -> Result<Vec<u8>, LocalOrRemoteFileReadError> {
        match self {
            LocalOrRemoteFile::Local(path) => {
                let mut file =
                    std::fs::File::open(path).map_err(LocalOrRemoteFileReadError::Local)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)
                    .map_err(LocalOrRemoteFileReadError::Local)?;
                Ok(buffer)
            }
            LocalOrRemoteFile::Remote(url) => {
                let response =
                    reqwest::blocking::get(url).map_err(LocalOrRemoteFileReadError::Remote)?;
                Ok(response
                    .bytes()
                    .map_err(LocalOrRemoteFileReadError::Remote)?
                    .to_vec())
            }
        }
    }

    pub async fn async_read_local_or_remote_file_to_bytes(
        self,
    ) -> Result<Vec<u8>, LocalOrRemoteFileReadError> {
        match self {
            LocalOrRemoteFile::Local(path) => {
                let mut file =
                    std::fs::File::open(path).map_err(LocalOrRemoteFileReadError::Local)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)
                    .map_err(LocalOrRemoteFileReadError::Local)?;
                Ok(buffer)
            }
            LocalOrRemoteFile::Remote(url) => {
                let response = reqwest::get(&url)
                    .await
                    .map_err(LocalOrRemoteFileReadError::Remote)?;
                Ok(response
                    .bytes()
                    .await
                    .map_err(LocalOrRemoteFileReadError::Remote)?
                    .to_vec())
            }
        }
    }
}

#[derive(Debug)]
pub enum LocalOrRemoteFileReadError {
    Local(std::io::Error),
    Remote(reqwest::Error),
}

// fn remote() -> UserDefinedEmbeddingModel {
//     // use the remote urls to create a UserDefinedEmbeddingModel
//     let user_defined_model = UserDefinedEmbeddingModel {
//         model_code: "user_defined_model".to_string(),
//         dim: 384,
//         description: "User defined model".to_string(),
//         onnx_file: read_local_or_remote_file_to_bytes(LocalOrRemoteFile::Remote(onnx_url)).unwrap(),
//         tokenizer_files: TokenizerFiles {
//             tokenizer_file: read_local_or_remote_file_to_bytes(LocalOrRemoteFile::Remote(
//                 tokenizer_url,
//             ))
//             .unwrap(),
//             config_file: read_local_or_remote_file_to_bytes(LocalOrRemoteFile::Remote(config_url))
//                 .unwrap(),
//             special_tokefuncns_map_file: read_local_or_remote_file_to_bytes(LocalOrRemoteFile::Remote(
//                 special_tokens_map_url,
//             ))
//             .unwrap(),
//             tokenizer_config_file: read_local_or_remote_file_to_bytes(LocalOrRemoteFile::Remote(
//                 tokenizer_config_url,
//             ))
//             .unwrap(),
//         },
//     };
//     user_defined_model
// }
