ARG RUST_VERSION=1.76.0

# Build stage
FROM rust:${RUST_VERSION}-slim-bookworm AS builder

# Install pkg-config and OpenSSL
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev

WORKDIR /app
COPY . .

RUN \
  --mount=type=cache,target=/app/target/ \
  --mount=type=cache,target=/usr/local/cargo/registry/ \
  cargo build --release && \
  cp ./target/release/fastembed-axum /

# Final stage
FROM debian:bookworm-slim AS final

# Create non-root user
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/nonexistent" \
  --shell "/sbin/nologin" \
  --no-create-home \
  --uid "10001" \
  appuser

# Copy the compiled binary
COPY --from=builder /fastembed-axum /usr/local/bin
RUN chown appuser /usr/local/bin/fastembed-axum

# Copy the configuration files
COPY --from=builder /app/config /opt/fastembed-axum/config
RUN chown -R appuser /opt/fastembed-axum

# Set user and environment variables
USER appuser
ENV RUST_LOG="hello_rs=debug,info"
WORKDIR /opt/fastembed-axum

# Set the entry point and expose port
ENTRYPOINT ["fastembed-axum"]
EXPOSE 8080/tcp
