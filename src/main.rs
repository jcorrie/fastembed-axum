use std::sync::{Arc, Mutex};

use aide::{
    axum::ApiRouter,
    openapi::{OpenApi, Tag},
    transform::TransformOpenApi,
};
use axum::{http::StatusCode, routing::get, Extension};
use fastembed::EmbeddingModel;
use listenfd::ListenFd;
use server::docs::docs_routes;
use server::errors::AppError;
use server::extractors::Json;
use server::state::AppState;
use tokio::net::TcpListener;
use uuid::Uuid;

pub mod embedding;
pub mod server;

#[tokio::main]
async fn main() {
    aide::gen::on_error(|error| {
        println!("{error}");
    });

    aide::gen::extract_schemas(true);
    let model: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;
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
        .route("/", get(hello_world))
        .nest_api_service("/embed", embedding::routes::embed_routes(state.clone()))
        .nest("/docs", docs_routes(state.clone()))
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

async fn hello_world() -> String {
    "Hello!".to_string()
}

fn api_docs(api: TransformOpenApi) -> TransformOpenApi {
    api.title("Fastembed axum server - API docs")
        .summary("Generate embeddings from text inputs using a rust implementation of fastembed.")
        .description(include_str!("README.md"))
        .tag(Tag {
            name: "embed".into(),
            description: Some("Generate embeddings".into()),
            ..Default::default()
        })
        .security_scheme(
            "ApiKey",
            aide::openapi::SecurityScheme::ApiKey {
                location: aide::openapi::ApiKeyLocation::Header,
                name: "X-Auth-Key".into(),
                description: Some("A key that is ignored.".into()),
                extensions: Default::default(),
            },
        )
        .default_response_with::<Json<AppError>, _>(|res| {
            res.example(AppError {
                error: "some error happened".to_string(),
                error_details: None,
                error_id: Uuid::nil(),
                // This is not visible.
                status: StatusCode::NOT_FOUND,
            })
        })
}
