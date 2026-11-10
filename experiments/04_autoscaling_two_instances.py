"""04. More users than one GPU can take: Cloud Run starts a second one.

A load generator keeps `users` requests in flight for `duration_s` seconds, more than one
instance accepts (`--concurrency` in gcp/deploy_vllm.sh). Requests that do not fit wait in
line, latency grows, and Cloud Run starts a second instance if `--max-instances` is at least
2 and the project has quota for a second L4. The second instance pays its own cold start
(experiment 01), so relief arrives minutes later, not seconds.

It records the latency of every request over time and tries two ways to know which instance
answered: a response header (stock vLLM sends none; the mock sends `x-instance-id`) and the
`process_start_time_seconds` of each /metrics scrape, which differs per instance.

    python experiments/04_autoscaling_two_instances.py --mock
    python experiments/04_autoscaling_two_instances.py --users 48 --duration 900   # live
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from vllm_lab import cli, prom

INSTANCE_HEADER = "x-instance-id"


def p95(values: list[float]) -> float | None:
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def one_request(base_url: str, i: int, max_tokens: int) -> tuple[float, str | None, bool]:
    body = {"model": "gemma", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": f"Pedido {i}: confirma la entrega."}]}
    req = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            r.read()
            return time.perf_counter() - t0, r.headers.get(INSTANCE_HEADER), True
    except (urllib.error.URLError, OSError):  # 429/503 when no instance can take it
        return time.perf_counter() - t0, None, False


def run(base_url: str, users: int = 24, duration_s: float = 4.0, max_tokens: int = 16,
        scrape_every_s: float = 0.25, bucket_s: float = 0.5) -> dict:
    records: list[dict] = []
    fingerprints: dict[float, float] = {}  # process_start_time -> first seen (s)
    lock = threading.Lock()
    t_start = time.perf_counter()
    stop_at = t_start + duration_s
    stop = threading.Event()

    def user(uid: int) -> None:
        n = 0
        while time.perf_counter() < stop_at:
            sent = time.perf_counter() - t_start
            lat, inst, ok = one_request(base_url, uid * 10_000 + n, max_tokens)
            with lock:
                records.append({"sent_s": round(sent, 3), "latency_s": round(lat, 4),
                                "instance": inst, "ok": ok})
            n += 1

    def scraper() -> None:
        while not stop.is_set():
            try:
                with urllib.request.urlopen(cli.root_of(base_url) + "/metrics", timeout=30) as r:
                    start = prom.gauge(prom.parse(r.read().decode()), ("process_start_time",))
                if start is not None:
                    fingerprints.setdefault(start, round(time.perf_counter() - t_start, 2))
            except (urllib.error.URLError, OSError):
                pass
            stop.wait(scrape_every_s)

    threads = [threading.Thread(target=user, args=(u,)) for u in range(users)]
    scr = threading.Thread(target=scraper, daemon=True)
    scr.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    records.sort(key=lambda r: r["sent_s"])

    # When did a second instance first answer?
    first_seen: dict[str, float] = {}
    for r in records:
        if r["instance"] and r["instance"] not in first_seen:
            first_seen[r["instance"]] = r["sent_s"]
    source = "header"
    if len(first_seen) < 2 and len(fingerprints) >= 2:
        source = "metrics process_start_time_seconds"
        first_seen = {f"process@{k:.0f}": v for k, v in fingerprints.items()}
    elif len(first_seen) < 2:
        source = "none (no header, one fingerprint)"
    second_at = sorted(first_seen.values())[1] if len(first_seen) >= 2 else None

    ok = [r for r in records if r["ok"]]
    before = [r["latency_s"] for r in ok if second_at is None or r["sent_s"] < second_at]
    after = [r["latency_s"] for r in ok if second_at is not None and r["sent_s"] >= second_at]
    p95_before, p95_after = p95(before), p95(after)

    timeline = []
    n_buckets = int(duration_s / bucket_s) + 1
    for b in range(n_buckets):
        lo, hi = b * bucket_s, (b + 1) * bucket_s
        rows = [r for r in ok if lo <= r["sent_s"] < hi]
        if not rows:
            continue
        lats = [r["latency_s"] for r in rows]
        by_inst: dict[str, int] = {}
        for r in rows:
            by_inst[r["instance"] or "?"] = by_inst.get(r["instance"] or "?", 0) + 1
        timeline.append({"t_s": round(lo, 2), "requests": len(rows),
                         "p50_s": round(statistics.median(lats), 3),
                         "p95_s": round(p95(lats), 3), "by_instance": by_inst})

    return {
        "users": users, "duration_s": duration_s, "max_tokens": max_tokens,
        "requests": len(records), "errors": len(records) - len(ok),
        "instances_seen": len(first_seen), "instance_source": source,
        "instance_first_seen_s": first_seen,
        "second_instance_at_s": second_at,
        "p95_before_s": _r(p95_before), "p95_after_s": _r(p95_after),
        "p95_reduction_x": (round(p95_before / p95_after, 2)
                            if p95_before and p95_after else None),
        "timeline": timeline,
    }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def main() -> None:
    def extra(p):
        p.add_argument("--users", type=int, default=None)
        p.add_argument("--duration", type=float, default=None)
        p.add_argument("--max-tokens", type=int, default=None)

    args = cli.parse(__doc__.splitlines()[0], extra)
    defaults = (24, 4.0, 16, 0.5) if args.mock else (48, 900.0, 128, 30.0)
    users = args.users or defaults[0]
    r = run(cli.resolve(args), users=users, duration_s=args.duration or defaults[1],
            max_tokens=args.max_tokens or defaults[2], bucket_s=defaults[3],
            scrape_every_s=0.25 if args.mock else 5.0)
    print(f"\n== {r['users']} users for {r['duration_s']} s: {r['requests']} requests, "
          f"{r['errors']} errors")
    for row in r["timeline"]:
        print(f"  t={row['t_s']:7.1f} s  n={row['requests']:4}  p50 {row['p50_s']:.3f} s  "
              f"p95 {row['p95_s']:.3f} s  {row['by_instance']}")
    cli.show("Second instance", {k: r[k] for k in (
        "instances_seen", "instance_source", "second_instance_at_s", "p95_before_s",
        "p95_after_s", "p95_reduction_x")})


if __name__ == "__main__":
    main()
