#!/usr/bin/env python3
"""Minimal OpenAI-compatible embedding server backed by SentenceTransformer on CPU."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("embedding-server")
MAX_REQUEST_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--served-model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--threads", default=8, type=int)
    return parser.parse_args()


class EmbeddingApplication:
    def __init__(self, model_path: Path, served_model_name: str, threads: int) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Embedding model directory does not exist: {model_path}")

        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, threads))
        self.model = SentenceTransformer(str(model_path.resolve()), device="cpu")
        self.model.max_seq_length = 256
        self.served_model_name = served_model_name
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.lock = threading.Lock()
        if self.dimension != 384:
            raise ValueError(
                f"Expected all-MiniLM-L6-v2 to produce 384 dimensions, got {self.dimension}"
            )
        LOGGER.info(
            "Loaded %s from %s on CPU (dimension=%d, max_seq_length=%d, threads=%d)",
            self.served_model_name,
            model_path.resolve(),
            self.dimension,
            self.model.max_seq_length,
            max(1, threads),
        )

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs or any(not isinstance(item, str) or not item.strip() for item in inputs):
            raise ValueError("input must contain one or more non-empty strings")
        with self.lock:
            vectors = self.model.encode(
                inputs,
                batch_size=min(32, len(inputs)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vectors.astype("float32", copy=False).tolist()


def make_handler(application: EmbeddingApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LiSSA-CPU-Embedding/1.0"

        def send_json(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/v1/health"}:
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model": application.served_model_name,
                        "device": "cpu",
                        "dimension": application.dimension,
                    },
                )
                return
            if self.path == "/v1/models":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": application.served_model_name,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/embeddings":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request body size")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                requested_model = request.get("model")
                if requested_model != application.served_model_name:
                    raise ValueError(
                        f"model must be {application.served_model_name!r}, got {requested_model!r}"
                    )
                if request.get("encoding_format", "float") != "float":
                    raise ValueError("only encoding_format='float' is supported")
                requested_dimensions = request.get("dimensions")
                if requested_dimensions not in {None, application.dimension}:
                    raise ValueError(f"dimensions must be {application.dimension}")

                raw_input = request.get("input")
                inputs = [raw_input] if isinstance(raw_input, str) else raw_input
                if not isinstance(inputs, list):
                    raise ValueError("input must be a string or a list of strings")
                vectors = application.embed(inputs)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "model": application.served_model_name,
                        "data": [
                            {"object": "embedding", "index": index, "embedding": vector}
                            for index, vector in enumerate(vectors)
                        ],
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    },
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                LOGGER.warning("Rejected embedding request: %s", exc)
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            except Exception as exc:  # pragma: no cover - defensive server boundary
                LOGGER.exception("Embedding request failed")
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": f"embedding failed: {type(exc).__name__}"}},
                )

        def log_message(self, format_string: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

    return Handler


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in the range 1..65535")
    if args.threads < 1:
        raise ValueError("threads must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    application = EmbeddingApplication(args.model_path, args.served_model_name, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))

    def stop_server(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    LOGGER.info("Listening on http://%s:%d/v1", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping embedding server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
