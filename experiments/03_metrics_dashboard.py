"""03. The six numbers worth watching on /metrics.

vLLM publishes hundreds of Prometheus series on /metrics. This script scrapes it before,
during and after a small burst of requests and prints only what you look at in production:
requests running and waiting, KV-cache usage, time to first token and end-to-end latency
(p50/p95 from the histograms, only for the burst), and generated tokens. Metric names change
between vLLM versions, so it matches them by fragments instead of exact names.

Behind Cloud Run each scrape lands on one instance; with several instances you see one of
them at a time.

    python experiments/03_metrics_dashboard.py [--mock]
"""

from __future__ import annotations

import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from vllm_lab import cli, make_client, prom, stream_chat

WATCH = {
    "requests_running": ("num_requests_running",),
    "requests_waiting": ("num_requests_waiting",),
    "kv_cache_usage": ("kv_cache_usage_perc", "gpu_cache_usage_perc", "cache_usage"),
}
HISTOGRAMS = {
    "ttft": ("time_to_first_token",),
    "e2e": ("e2e_request_latency",),
}
GENERATED = ("generation_tokens",)
PROMPT = ("prompt_tokens",)


def scrape(base_url: str) -> list[prom.Sample]:
    with urllib.request.urlopen(cli.root_of(base_url) + "/metrics", timeout=60) as r:
        return prom.parse(r.read().decode())


def snapshot(samples: list[prom.Sample]) -> dict:
    out = {k: prom.gauge(samples, frags) for k, frags in WATCH.items()}
    out["generation_tokens_total"] = prom.gauge(samples, GENERATED, "_total")
    out["prompt_tokens_total"] = prom.gauge(samples, PROMPT, "_total")
    return out


def run(base_url: str, users: int = 8, max_tokens: int = 64) -> dict:
    before = scrape(base_url)
    client = make_client(base_url)
    peak: dict = {"requests_running": 0.0, "requests_waiting": 0.0, "kv_cache_usage": 0.0}
    done = threading.Event()

    def watch() -> None:
        while not done.is_set():
            snap = snapshot(scrape(base_url))
            for k, seen in list(peak.items()):
                peak[k] = max(seen, snap.get(k) or 0.0)
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    prompts = [f"Cliente {i}: resume en tres frases el estado de tu pedido {i}." for i in
               range(users)]
    with ThreadPoolExecutor(max_workers=users) as pool:
        list(pool.map(lambda p: stream_chat(client, p, max_tokens=max_tokens, ignore_eos=True),
                      prompts))
    done.set()
    watcher.join()
    after = scrape(base_url)

    latency = {}
    for key, frags in HISTOGRAMS.items():
        h = prom.diff_buckets(prom.buckets(after, frags), prom.buckets(before, frags))
        latency[key] = {"name": prom.find_name(after, frags, "_bucket"),
                        "requests": h[-1][1] if h else 0,
                        "p50_s": _r(prom.quantile(0.50, h)), "p95_s": _r(prom.quantile(0.95, h))}
    a, b = snapshot(after), snapshot(before)
    names = {k: prom.find_name(after, frags) for k, frags in WATCH.items()}
    names["generation_tokens"] = prom.find_name(after, GENERATED, "_total")
    gen_delta = (a["generation_tokens_total"] or 0) - (b["generation_tokens_total"] or 0)
    return {
        "series_total": len(after),
        "matched_names": names,
        "peak_during_burst": peak,
        "after_burst": {k: a[k] for k in WATCH},
        "latency": latency,
        "generation_tokens_in_burst": gen_delta,
        "expected_generation_tokens": users * max_tokens,
        "parsed_all": all(v is not None for v in names.values())
        and all(v["p95_s"] is not None for v in latency.values()),
    }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    r = run(cli.resolve(args))
    print(f"\n== /metrics: {r['series_total']} series, these are the ones that matter")
    for k, v in r["peak_during_burst"].items():
        print(f"  {k:22} peak {v:8.3f}   ({r['matched_names'][k]})")
    for k, v in r["latency"].items():
        print(f"  {k:22} p50 {v['p50_s']} s  p95 {v['p95_s']} s  over {v['requests']:.0f} "
              f"requests  ({v['name']})")
    print(f"  {'generated tokens':22} {r['generation_tokens_in_burst']:.0f} in the burst "
          f"({r['matched_names']['generation_tokens']})")


if __name__ == "__main__":
    main()
