#!/usr/bin/env python3
"""Check whether an exact model id is ready on a local OpenAI-compatible API."""

from __future__ import annotations

import argparse
import http.client
import json
from urllib.parse import urlparse


MAX_RESPONSE_BYTES = 1024 * 1024


def model_is_ready(base_url: str, expected_model: str, timeout: float) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        return False

    port = parsed.port or 80
    models_path = f"{parsed.path.rstrip('/')}/models" or "/models"
    connection = http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)
    try:
        connection.request("GET", models_path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            return False
        raw_payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError, http.client.HTTPException):
        return False
    finally:
        connection.close()

    if len(raw_payload) > MAX_RESPONSE_BYTES:
        return False
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    models = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(models, list) and any(
        isinstance(item, dict) and item.get("id") == expected_model for item in models
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return 0 if model_is_ready(
        arguments.base_url, arguments.expected_model, arguments.timeout
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
