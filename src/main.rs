use std::sync::Arc;

use aide::{
    axum::ApiRouter,
    openapi::{OpenApi, Tag},
    transform::TransformOpenApi,
};
use axum::{http::StatusCode, routing::get, Extension};
use docs::docs_routes;
use errors::AppError;
use extractors::Json;
use listenfd::ListenFd;
use state::AppState;
use tokio::net::TcpListener;
use uuid::Uuid;

pub mod docs;
pub mod embedding;
pub mod errors;
pub mod extractors;
pub mod state;

#[tokio::main]
async fn main() {
    aide::gen::on_error(|error| {
        println!("{error}");
    });

    aide::gen::extract_schemas(true);
    let model = embedding::init_model();
    let state = AppState {
        model: Arc::new(model),
    };

    let mut api = OpenApi::default();

    let app = ApiRouter::new()
        .route("/", get(hello_world))
        .nest_api_service("/embed", embedding::routes::embed_routes(state.clone()))
        .nest_api_service("/docs", docs_routes(state.clone()))
        .finish_api_with(&mut api, api_docs)
        .layer(Extension(Arc::new(api))) // Arc is very important here or you will face massive memory and performance issues
        .with_state(state);

    println!("Example docs are accessible at http://127.0.0.1:3100/docs");

    let mut listenfd = ListenFd::from_env();
    // let listener = TcpListener::bind("0.0.0.0:3100").await.unwrap();
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
    api.title("Aide axum Open API")
        .summary("An example Todo application")
        .description(include_str!("README.md"))
        .tag(Tag {
            name: "todo".into(),
            description: Some("Todo Management".into()),
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
                status: StatusCode::IM_A_TEAPOT,
            })
        })
}
