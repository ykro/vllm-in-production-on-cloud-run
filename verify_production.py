"""The Outside Judge for vLLM in production on Cloud Run: runs the five experiments once and checks what must hold.

    python verify_production.py          # live vLLM on Cloud Run through the tunnel
    python verify_production.py --mock   # fake service, checks the logic without a GPU
    python verify_production.py --from-saved   # re-reads data/startup_log.txt, reuses 02-05 from
                                               # benchmarks/latest_results.json (no servers)

The checks are about behavior (the cold start phases add up, the server is ready, /metrics
can be read, a second instance lowers p95 under load, the break-even exists), not about
exact numbers. In live mode it also saves benchmarks/latest_results.json and
benchmarks/terminal_output.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vllm_lab.client import BASE_URL

EXPERIMENTS = [
    "01_cold_start_anatomy",
    "02_health_and_readiness",
    "03_metrics_dashboard",
    "04_autoscaling_two_instances",
    "05_cost_vs_api",
]
LIVE_LOG = ROOT / "data" / "startup_log.txt"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--startup-log", help="defaults to data/startup_log.txt when it exists")
    p.add_argument("--compare-log", help="second startup log (e.g. weights from the bucket)")
    p.add_argument("--users", type=int, help="load for experiment 04 (live default 48)")
    p.add_argument("--duration", type=float, help="seconds of load in 04 (live default 900)")
    p.add_argument("--from-saved", action="store_true",
                   help="rebuild the report from benchmarks/latest_results.json (no servers); "
                        "experiment 01 is recomputed from the startup log")
    p.add_argument("--verbose", action="store_true", help="print each experiment's raw result")
    args = p.parse_args()

    from vllm_lab import cli

    log = args.startup_log or (LIVE_LOG if LIVE_LOG.exists() and not args.mock else None)
    mods = {name: load(name) for name in EXPERIMENTS}
    if args.from_saved:
        saved = json.loads((ROOT / "benchmarks" / "latest_results.json").read_text())
        res = dict(saved["experiments"])
        res[EXPERIMENTS[0]] = mods[EXPERIMENTS[0]].run(log, args.compare_log)
        cold = res[EXPERIMENTS[0]]["total_s"]
        # 02 and 05 depend on the cold start and the idle time: recompute them from the logs.
        res[EXPERIMENTS[1]]["startup_probe"] = mods[EXPERIMENTS[1]].probe_budget(cold_start_s=cold)
        res[EXPERIMENTS[4]] = mods[EXPERIMENTS[4]].refresh_idle(res[EXPERIMENTS[4]], cold)
        base_url = None
    else:
        base_url = cli.resolve(args)
        r01 = mods[EXPERIMENTS[0]].run(log, args.compare_log)
        res = {
            EXPERIMENTS[0]: r01,
            EXPERIMENTS[1]: mods[EXPERIMENTS[1]].run(base_url, cold_start_s=r01["total_s"]),
            EXPERIMENTS[2]: mods[EXPERIMENTS[2]].run(base_url),
            EXPERIMENTS[3]: (mods[EXPERIMENTS[3]].run(base_url, users=args.users or 24,
                                                      duration_s=args.duration or 4.0)
                             if args.mock else
                             mods[EXPERIMENTS[3]].run(base_url, users=args.users or 48,
                                                      duration_s=args.duration or 900.0,
                                                      max_tokens=128, scrape_every_s=5.0,
                                                      bucket_s=30.0)),
        }
    r1 = res[EXPERIMENTS[0]]
    cold = r1["total_s"]
    if not args.from_saved:
        res[EXPERIMENTS[4]] = mods[EXPERIMENTS[4]].run(base_url, cold_start_s=cold)
    r2, r3, r4, r5 = (res[n] for n in EXPERIMENTS[1:])
    if args.verbose:
        for name, value in res.items():
            cli.show(name, value)

    s1 = r1["summary"]
    lat = r3["latency"]
    cur = r5["current"]
    one, many = r5["rows"][0], r5["rows"][-1]
    be = cur["breakeven_tokens_per_day"]
    rows = [
        ("01. Cold start anatomy",
         f"{s1['total_min']} min: weights {s1['weights_s']:.0f} s, compile {s1['compile_s']:.0f} s",
         f"graphs {s1['cuda_graphs_s']:.0f} s",
         abs(r1["phases_sum_s"] - r1["total_s"]) < 1e-6 and s1["weights_s"] > 0),
        ("02. Health and readiness",
         f"/health {r2['health']['status']}, 1 token in {r2['readiness']['ms']:.0f} ms",
         f"probe budget {r2['startup_probe']['budget_s']} s",
         r2["health"]["status"] == 200 and r2["readiness"]["ready"]
         and r2["startup_probe"]["fits"]),
        ("03. /metrics that matter",
         f"TTFT p95 {lat['ttft']['p95_s']} s, e2e p95 {lat['e2e']['p95_s']} s",
         f"{r3['generation_tokens_in_burst']:.0f} tokens",
         r3["parsed_all"] and r3["generation_tokens_in_burst"] > 0),
        ("04. Second instance under load",
         (f"{r4['instances_seen']} instances, 2nd at {r4['second_instance_at_s']} s"
          if r4["second_instance_at_s"] is not None else "one instance only"),
         (f"p95 {r4['p95_before_s']}->{r4['p95_after_s']} s"
          if r4["p95_after_s"] is not None else "no p95 after"),
         r4["instances_seen"] >= 2 and r4["p95_after_s"] is not None
         and r4["p95_after_s"] < r4["p95_before_s"]),
        ("05. Cost vs hosted API",
         (f"${one['usd_per_million_output_tokens']:.2f} (1u) "
          f"${many['usd_per_million_output_tokens']:.2f} ({many['users']}u) /1M"),
         f"break-even {be:,} tok/day",
         isinstance(be, int) and be > 0 and math.isfinite(be)
         and many["output_tokens_per_s"] > one["output_tokens_per_s"]),
    ]

    mode = "mock (fake service, numbers are not real)" if args.mock else "live"
    lines = [
        "=" * 114,
        "THE OUTSIDE JUDGE — vLLM in Production on Cloud Run",
        "=" * 114,
        f"Mode    : {mode}",
        (f"Server  : vLLM {r2['version']['vllm']} | model {r2['models']['weights']} served as "
         f"'{r2['models']['id']}'"),
        f"Startup : {r1['log']}{' (sample written from the measured run)' if r1['is_sample'] else ''}",
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 114,
        f"{'Experiment':32} | {'Measured':44} | {'Result':28} | Check",
        "-" * 114,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:32} | {measured:44} | {result:28} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 114,
        (f"Instance: {r5['instance']}, ${r5['usd_per_hour']}/h list price checked "
         f"{r5['price_checked']}."),
        (f"API     : {r5['api_model']}, ${cur['api_usd_per_million_output_blended']} per 1M "
         f"output with {r5['input_per_output']} input tokens each, checked {r5['api_checked']}."),
        (f"Idle    : min-instances 1 ${cur['idle']['min_instances_1_usd_per_month']}/month; "
         f"min-instances 0 $0 idle, ${cur['idle']['min_instances_0_usd_per_wake']} per wake."),
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": (saved["generated_at"] if args.from_saved
                             else datetime.now(UTC).isoformat(timespec="seconds")),
            "vllm_version": r2["version"]["vllm"],
            "experiments": res,
            "checks": {name: ok for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
