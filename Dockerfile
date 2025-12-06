# STAGE 1 
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
COPY src ./src 

ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RUN cargo build --release 

# STAGE 2
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

RUN python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install --no-cache-dir -r nexum_ai/requirements.txt

RUN useradd --system --create-home --home-dir /app --shell /bin/bash nexumuser && \
	chown -R nexumuser:nexumuser /app

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

USER nexumuser

CMD ["nexum"]
