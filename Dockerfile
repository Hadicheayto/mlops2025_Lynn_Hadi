FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y make curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Copy everything first (important)
COPY . .

# Create venv and install deps
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync

# Install package explicitly
RUN . .venv/bin/activate && pip install -e .

CMD . .venv/bin/activate && make full
