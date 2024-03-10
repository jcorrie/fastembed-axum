use std::sync::Arc;

use aide::{
    axum::{
        routing::{get, get_with},
        ApiRouter, IntoApiResponse,
    },
    openapi::OpenApi,
    redoc::Redoc,
    scalar::Scalar,
};

use aide::{openapi::Tag, transform::TransformOpenApi};

use axum::{response::IntoResponse, Extension};

use crate::server::errors::AppError;
use crate::server::extractors::Json;
use uuid::Uuid;

use axum::http::StatusCode;

use super::state::AppState;

pub fn docs_routes(state: AppState) -> ApiRouter {
    aide::gen::infer_responses(true);

    let router: ApiRouter = ApiRouter::new()
        .api_route_with(
            "/",
            get_with(
                Scalar::new("/docs/private/api.json")
                    .with_title("Aide Axum")
                    .axum_handler(),
                |op| op.description("This documentation page."),
            ),
            |p| p.security_requirement("ApiKey"),
        )
        .api_route_with(
            "/redoc",
            get_with(
                Redoc::new("/docs/private/api.json")
                    .with_title("Aide Axum")
                    .axum_handler(),
                |op| op.description("This documentation page."),
            ),
            |p| p.security_requirement("ApiKey"),
        )
        .route("/private/api.json", get(serve_docs))
        .with_state(state);

    // Afterwards we disable response inference because
    // it might be incorrect for other routes.
    aide::gen::infer_responses(false);

    router
}

async fn serve_docs(Extension(api): Extension<Arc<OpenApi>>) -> impl IntoApiResponse {
    Json(api).into_response()
}

pub fn api_docs(api: TransformOpenApi) -> TransformOpenApi {
    api.title("Fastembed axum server - API docs")
        .summary("Generate embeddings from text inputs using a rust implementation of fastembed.")
        .description(include_str!("../embedding/README.md"))
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
