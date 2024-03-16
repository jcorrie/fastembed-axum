pub mod embedding;
pub mod server;

fn main() {
    server::run::start_server(None, embedding::ModelSource::HuggingFace);
}
