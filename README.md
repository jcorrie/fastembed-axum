# Web server for sentence embeddings

A small axum server exposing functionality from the fastembed-rs crate

Run with `cargo run`

Run with auto reload for development


```sh
cargo install cargo-watch systemfd
```

```sh
systemfd --no-pid -s http::3100 -- cargo watch -x run
```

Visit docs at `localhost:3100/docs`