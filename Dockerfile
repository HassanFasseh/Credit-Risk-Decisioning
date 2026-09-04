# Serves a pre-trained model. Does not train inside the container -- run
# `python src/train_final_model.py` locally first so artifacts/ exists; this
# build copies that output in. Training needs the full raw dataset (~2GB,
# gitignored); the served image should not need to carry it.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

FROM python:3.12-slim
# lightgbm's Linux wheel links against libgomp at runtime; python:3.12-slim
# doesn't ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY artifacts/ ./artifacts/

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
