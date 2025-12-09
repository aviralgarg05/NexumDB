##################################################
#################### STAGE 1 #####################
##################################################
FROM rust:1-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
	python3 \
	python3-dev \
	build-essential \
	libssl-dev \
	pkg-config \
	ca-certificates \
	libclang-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY nexum_cli/Cargo.toml ./nexum_cli/
COPY nexum_core/Cargo.toml ./nexum_core/
COPY tests/Cargo.toml ./tests/

RUN mkdir -p nexum_core/src && echo "" > nexum_core/src/lib.rs
RUN mkdir -p nexum_cli/src && echo "fn main() {}" > nexum_cli/src/main.rs
RUN mkdir -p tests/src && echo "" > tests/src/lib.rs

ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
RUN cargo build --release --workspace

COPY nexum_core ./nexum_core
COPY nexum_cli ./nexum_cli
COPY tests ./tests

RUN cargo test --release --workspace
RUN touch nexum_cli/src/main.rs && cargo build --release --workspace --exclude tests


##################################################
#################### STAGE 2 #####################
##################################################
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
	python3 \
	python3-venv \
	python3-dev \
	libssl-dev \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/nexum /usr/local/bin/nexum
COPY nexum_ai ./nexum_ai

RUN useradd --system --create-home --home-dir /app --shell /bin/bash nexumuser && \
	chown -R nexumuser:nexumuser /app

USER nexumuser

RUN python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
	pip install --no-cache-dir -r nexum_ai/requirements.txt

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

CMD ["nexum"]
