"""02. Alive, loaded and ready are three different answers.

Asks the server the three questions an operator asks: is the process up (`/health`), which
model and version is it serving (`/v1/models`, `/version`), and can it actually generate
(one tiny completion of a single token). Then explains the Cloud Run startup probe: Cloud Run
sends no traffic to an instance until `/health` answers, and gives up after
periodSeconds x failureThreshold seconds. That budget has to be larger than your slowest
cold start, or Cloud Run kills the instance while it is still downloading weights.

    python experiments/02_health_and_readiness.py [--mock]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from vllm_lab import cli

# Startup probe used by gcp/deploy_vllm.sh
PERIOD_S = 10
FAILURE_THRESHOLD = 150
TIMEOUT_S = 5
# Measured cold start with weights mounted from a bucket (experiment 01): 580 s, 9.7 min.
MEASURED_COLD_START_S = 580


def timed_get(url: str, timeout: float = 60) -> dict:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode() or ""
            status = r.status
    except urllib.error.HTTPError as exc:
        body, status = exc.read().decode(), exc.code
    return {"status": status, "ms": round((time.perf_counter() - t0) * 1000, 1), "body": body}


def ready(base_url: str, model: str) -> dict:
    """Readiness that means something: the engine produces one token end to end."""
    body = {"model": model, "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        ok = out["usage"]["completion_tokens"] >= 1
        return {"ready": ok, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "completion_tokens": out["usage"]["completion_tokens"]}
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        return {"ready": False, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": str(exc)}


def probe_budget(period_s: int = PERIOD_S, failure_threshold: int = FAILURE_THRESHOLD,
                 cold_start_s: float = MEASURED_COLD_START_S) -> dict:
    budget = period_s * failure_threshold
    return {
        "flag": (f"--startup-probe httpGet.path=/health,httpGet.port=8080,"
                 f"periodSeconds={period_s},failureThreshold={failure_threshold},"
                 f"timeoutSeconds={TIMEOUT_S}"),
        "budget_s": budget,
        "measured_cold_start_s": cold_start_s,
        "headroom_x": round(budget / cold_start_s, 2),
        "fits": budget > cold_start_s,
        "probes_until_ready": -(-int(cold_start_s) // period_s),
        "rule": "budget >= 2x the slowest cold start you have measured",
    }


def run(base_url: str) -> dict:
    root = cli.root_of(base_url)
    health = timed_get(root + "/health")
    models = timed_get(base_url + "/models")
    version = timed_get(root + "/version")
    model = json.loads(models["body"])["data"][0] if models["status"] == 200 else {}
    try:
        vllm_version = json.loads(version["body"]).get("version", "?")
    except ValueError:
        vllm_version = "?"
    return {
        "health": {"status": health["status"], "ms": health["ms"]},
        "models": {"status": models["status"], "ms": models["ms"], "id": model.get("id"),
                   "weights": model.get("root"), "max_model_len": model.get("max_model_len")},
        "version": {"status": version["status"], "ms": version["ms"], "vllm": vllm_version},
        "readiness": ready(base_url, model.get("id", "gemma")),
        "startup_probe": probe_budget(),
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    r = run(cli.resolve(args))
    for key in ("health", "models", "version", "readiness", "startup_probe"):
        cli.show(key, r[key])


if __name__ == "__main__":
    main()
