use std::sync::{Arc, Mutex};

use crate::embedding;
use crate::server::docs::{api_docs, docs_routes};

use crate::server::state::AppState;
use aide::{axum::ApiRouter, openapi::OpenApi};
use axum::{routing::get, Extension};
use fastembed::EmbeddingModel;
use listenfd::ListenFd;
use tokio::net::TcpListener;

#[tokio::main]
pub async fn start_server() {
    aide::gen::on_error(|error| {
        println!("{error}");
    });

    aide::gen::extract_schemas(true);
    let model: EmbeddingModel = EmbeddingModel::BGEBaseENV15;
    let model_info: embedding::JSONModelInfo =
        embedding::get_current_model_info(&model).expect("Can't load model");
    let text_embedding = embedding::new_text_embedding(&model);
    let state = AppState {
        text_embedding: Arc::new(text_embedding),
        model: Arc::new(Mutex::new(model)),
        model_info,
    };

    let mut api = OpenApi::default();

    let app = ApiRouter::new()
        .route("/", get(embedding::routes::hello_world))
        .nest_api_service("/embed", embedding::routes::embed_routes(state.clone()))
        .nest("/test", docs_routes(state.clone(), Some("/test")))
        .finish_api_with(&mut api, api_docs)
        .layer(Extension(Arc::new(api)));

    println!("Example docs are accessible at http://127.0.0.1:3100/docs");

    let mut listenfd = ListenFd::from_env();
    let listener = match listenfd.take_tcp_listener(0).unwrap() {
        // if we are given a tcp listener on listen fd 0, we use that one
        Some(listener) => {
            listener.set_nonblocking(true).unwrap();
            TcpListener::from_std(listener).unwrap()
        }
        // otherwise fall back to local listening
        None => TcpListener::bind("127.0.0.1:3100").await.unwrap(),
    };

    axum::serve(listener, app).await.unwrap();
}
