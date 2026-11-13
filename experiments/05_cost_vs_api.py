"""05. What a million tokens really costs, and when a hosted API is cheaper.

Measures output tokens per second with 1 user and with N users, turns them into dollars per
million output tokens with the Cloud Run list price (benchmarks/pricing.json), and compares
with the list price of a hosted Gemini Flash-class API (benchmarks/api_prices.json).

Then the part a per-token price hides:
- min-instances 1 bills the GPU 24 hours a day, busy or idle;
- min-instances 0 bills nothing while idle, but every wake-up pays a cold start plus the
  idle time Cloud Run keeps the instance before scaling it down (measured from
  data/instance_lifecycle_log.txt: startup probe passed -> "Shutting down" on a revision with no traffic);
- break-even: how many output tokens per day you need before the always-on GPU costs less
  than the API, and whether one instance can even produce that many.

    python experiments/05_cost_vs_api.py [--mock]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from vllm_lab import cli, make_client, stream_chat

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
PRICING = BENCH / "pricing.json"
API_PRICES = BENCH / "api_prices.json"
HOURS_PER_MONTH = 730
# Input tokens per output token in a typical app request (prompt + context vs answer).
# The API bills input too; on your GPU prefill is cheap and already inside the hourly price.
INPUT_PER_OUTPUT = 2.0
COLD_START_S = 580  # fallback; verify_production.py passes what experiment 01 measures
# Cloud Logging export of instance starts, startup probes and shutdowns on 2026-11-11.
LIFECYCLE_LOG = ROOT / "data" / "instance_lifecycle_log.txt"
# The min-instances 0 revision deployed by `make cool`: it got no request, so the time between its
# startup probe and its shutdown is how long Cloud Run kept an idle GPU instance alive.
IDLE_REVISION = "vllm-gemma-00010-bmh"


def idle_before_scale_down(log_path: Path = LIFECYCLE_LOG, revision: str = IDLE_REVISION) -> dict:
    """Seconds an instance with no traffic stayed up: startup probe passed -> 'Shutting down'."""
    probe = shutdown = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stamp, rev, text = line.split("\t", 2)
        if rev != revision:
            continue
        if text.startswith("STARTUP HTTP probe succeeded"):
            probe = datetime.fromisoformat(stamp)
        elif "Shutting down" in text:
            shutdown = datetime.fromisoformat(stamp) if shutdown is None else min(shutdown, datetime.fromisoformat(stamp))
    if probe is None or shutdown is None or shutdown <= probe:
        raise ValueError(f"{log_path.name}: no probe -> shutdown pair for {revision}")
    return {"log": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
            "revision": revision,
            "probe_succeeded_at": probe.isoformat(), "shutting_down_at": shutdown.isoformat(),
            "seconds": round((shutdown - probe).total_seconds(), 1)}


def wake_usd(per_hour: float, cold_start_s: float, idle_tail_s: float) -> float:
    """What one wake-up of a min-instances 0 service costs: cold start + idle time before scale-down."""
    return round((cold_start_s + idle_tail_s) * per_hour / 3600, 3)


def throughput(base_url: str, users: int, out_tokens: int) -> float:
    client = make_client(base_url)
    prompts = [f"Usuario {i}: redacta el resumen semanal del pedido {i}." for i in range(users)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as pool:
        results = list(pool.map(lambda p: stream_chat(client, p, max_tokens=out_tokens,
                                                      ignore_eos=True), prompts))
    return sum(r.completion_tokens for r in results) / (time.perf_counter() - t0)


def api_usd_per_million_output(model: dict, input_per_output: float) -> float:
    """Blended API price per 1M output tokens, counting the input tokens that come with them."""
    return model["usd_per_million_output"] + input_per_output * model["usd_per_million_input"]


def economics(per_hour: float, tps_by_users: dict[int, float], api_blended: float,
              cold_start_s: float, idle_tail_s: float) -> dict:
    rows = [{"users": u, "output_tokens_per_s": round(tps, 1),
             "usd_per_million_output_tokens": round(per_hour / (tps * 3600) * 1e6, 3),
             "tokens_per_day_at_full_load": round(tps * 86400)}
            for u, tps in sorted(tps_by_users.items())]
    best = rows[-1]
    always_on_day = per_hour * 24
    breakeven = always_on_day / api_blended * 1e6
    capacity = best["tokens_per_day_at_full_load"]
    return {
        "rows": rows,
        "api_usd_per_million_output_blended": round(api_blended, 3),
        "idle": {
            "min_instances_0_usd_per_month_idle": 0.0,
            "min_instances_1_usd_per_month": round(per_hour * HOURS_PER_MONTH, 2),
            "min_instances_0_usd_per_wake": wake_usd(per_hour, cold_start_s, idle_tail_s),
            "wake_assumes_s": {"cold_start": cold_start_s, "idle_before_scale_down": idle_tail_s},
        },
        "breakeven_tokens_per_day": round(breakeven),
        "breakeven_vs_capacity": round(breakeven / capacity, 3),
        "breakeven_reachable": breakeven <= capacity,
        "gpu_cheaper_per_token_at_full_load": best["usd_per_million_output_tokens"] < api_blended,
    }


def run(base_url: str, users: int = 8, out_tokens: int = 256,
        input_per_output: float = INPUT_PER_OUTPUT, cold_start_s: float = COLD_START_S) -> dict:
    price = json.loads(PRICING.read_text())
    api = json.loads(API_PRICES.read_text())
    per_hour = price["usd_per_hour"]
    idle = idle_before_scale_down()
    stream_chat(make_client(base_url), "Hola", max_tokens=8)  # warmup
    tps = {u: throughput(base_url, u, out_tokens) for u in (1, users)}
    periods = {}
    for period in api["periods"]:
        blended = api_usd_per_million_output(period, input_per_output)
        periods[period["label"]] = economics(per_hour, tps, blended, cold_start_s, idle["seconds"])
    current = periods[api["periods"][0]["label"]]
    rows = current["rows"]
    periods = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in periods.items()}
    return {
        "instance": price["instance"], "usd_per_hour": per_hour,
        "price_checked": price["checked"],
        "api_model": api["model"], "api_checked": api["checked"], "api_source": api["source"],
        "input_per_output": input_per_output,
        "users_compared": [1, users],
        "idle_before_scale_down": idle,
        "rows": rows,
        "current": periods[api["periods"][0]["label"]],
        "by_api_period": periods,
    }


def refresh_idle(result: dict, cold_start_s: float, idle: dict | None = None) -> dict:
    """Recomputes the wake-up cost of a saved result with the measured cold start and idle time."""
    idle = idle or idle_before_scale_down()
    result["idle_before_scale_down"] = idle
    for period in result["by_api_period"].values():
        period["idle"]["min_instances_0_usd_per_wake"] = wake_usd(result["usd_per_hour"],
                                                                  cold_start_s, idle["seconds"])
        period["idle"]["wake_assumes_s"] = {"cold_start": cold_start_s,
                                            "idle_before_scale_down": idle["seconds"]}
    result["current"] = result["by_api_period"][next(iter(result["by_api_period"]))]
    return result


def main() -> None:
    def extra(p):
        p.add_argument("--users", type=int, default=8)
        p.add_argument("--input-per-output", type=float, default=INPUT_PER_OUTPUT)

    args = cli.parse(__doc__.splitlines()[0], extra)
    r = run(cli.resolve(args), users=args.users, input_per_output=args.input_per_output)
    print(f"\n== {r['instance']} at ${r['usd_per_hour']}/h vs {r['api_model']}")
    for row in r["rows"]:
        print(f"  {row['users']:3} users  {row['output_tokens_per_s']:8.1f} tok/s  "
              f"${row['usd_per_million_output_tokens']:.3f} per 1M output tokens")
    for label, e in r["by_api_period"].items():
        print(f"\n  API price {label}: ${e['api_usd_per_million_output_blended']} per 1M output "
              f"(with {r['input_per_output']} input tokens each)")
        print(f"    min-instances 1: ${e['idle']['min_instances_1_usd_per_month']}/month; "
              f"min-instances 0: $0 idle, ${e['idle']['min_instances_0_usd_per_wake']} per wake")
        print(f"    break-even: {e['breakeven_tokens_per_day']:,} output tokens/day "
              f"({e['breakeven_vs_capacity']:.0%} of one instance at full load)")


if __name__ == "__main__":
    main()
