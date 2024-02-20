use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate fastembed_axum;
use fastembed_axum::embedding::embed_documents;
use fastembed_axum::embedding::{EmbeddingRequestUnit, EmbeddingResponse};

fn main_embed_bench(docs: &Vec<String>) -> EmbeddingResponse {
    let model = fastembed_axum::embedding::init_text_embedding();
    let request_objects: Vec<EmbeddingRequestUnit> = docs
        .iter()
        .enumerate()
        .map(|(i, text)| EmbeddingRequestUnit {
            text_to_embed: text.to_string(),
            id: i as i32,
        })
        .collect();
    embed_documents(&model, request_objects)
}

fn criterion_benchmark(c: &mut Criterion) {
    let file_content = include_str!("../benches/input.txt");
    let bench_docs: Vec<String> = file_content.lines().map(|x| x.to_string()).collect();

    c.bench_function("embed docs", |b| {
        b.iter(|| main_embed_bench(black_box(&bench_docs)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
