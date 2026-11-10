"""Shared command line for the experiments: live server by default, --mock without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

from vllm_lab.client import BASE_URL

TUNNEL_HINT = ("Open the tunnel first:  gcloud run services proxy vllm-gemma "
               "--region us-east4 --port 8000\nor run with --mock.")


def parse(description: str,
          extra: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mock", action="store_true", help="use the built-in fake service (no GPU)")
    p.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible /v1 URL")
    if extra:
        extra(p)
    return p.parse_args()


def root_of(base_url: str) -> str:
    """http://localhost:8000/v1 -> http://localhost:8000 (where /health and /metrics live)."""
    return base_url.rsplit("/v1", 1)[0]


def resolve(args: argparse.Namespace) -> str:
    if args.mock:
        from vllm_lab import mock_server

        _, url = mock_server.start()
        return url
    try:
        # The first request through the tunnel can wake a scaled-to-zero service: be patient.
        urllib.request.urlopen(root_of(args.base_url) + "/health", timeout=900).read()
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"No vLLM server at {root_of(args.base_url)} ({exc}).\n{TUNNEL_HINT}")
    return args.base_url


def show(title: str, payload: dict) -> None:
    print(f"\n== {title}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
