use std::sync::{Arc, Mutex};

use crate::embedding::{self, HFEmbeddingModelOrUserDefinedModel};
use crate::server::docs::{api_docs, docs_routes};

use crate::server::state::{get_app_state, AppState};
use aide::{axum::ApiRouter, openapi::OpenApi};
use axum::{routing::get, Extension};
use listenfd::ListenFd;
use tokio::net::TcpListener;

const DEFAULT_BASE_API_URL: &str = "";

#[tokio::main]
pub async fn start_server(api_base_url: Option<&str>, model_source: embedding::ModelSource) {
    aide::gen::on_error(|error| {
        println!("{error}");
    });
    let base_api_url = api_base_url.unwrap_or(DEFAULT_BASE_API_URL);
    aide::gen::extract_schemas(true);

    let mut api = OpenApi::default();
    let state = get_app_state(model_source);
    let app = ApiRouter::new()
        .route(
            &base_api_route_builder("/", base_api_url),
            get(embedding::routes::hello_world),
        )
        .nest_api_service(
            &base_api_route_builder("/embed", base_api_url),
            embedding::routes::embed_routes(state.clone()),
        )
        .nest(
            &base_api_route_builder("/docs", base_api_url),
            docs_routes(
                state.clone(),
                Some(&base_api_route_builder("/docs", base_api_url)),
            ),
        )
        .finish_api_with(&mut api, api_docs)
        .layer(Extension(Arc::new(api)));

    println!(
        "Example docs are accessible at http://127.0.0.1:3100{}",
        base_api_route_builder("/docs", base_api_url)
    );

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

fn base_api_route_builder(endpoint: &str, api_base_url: &str) -> String {
    format!("{}{}", api_base_url, endpoint)
}
