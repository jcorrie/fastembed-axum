# Web server for sentence embeddings

A small axum server exposing functionality from the [fastembed-rs crate](https://github.com/Anush008/fastembed-rs)

Run with `cargo run`

Run with auto reload for development

```sh
cargo install cargo-watch systemfd
systemfd --no-pid -s http::3100 -- cargo watch -x run
```

OpenAPI documentation created with [aide](https://github.com/tamasfe/aide). Visit at: `localhost:3100/docs`