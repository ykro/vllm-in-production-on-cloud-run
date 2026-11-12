"""01. Where eight and a half minutes of cold start go.

Reads a vLLM startup log exported from Cloud Logging and splits the cold start into phases:
container start, process init, downloading weights, loading them into the GPU, torch.compile,
KV-cache profiling, CUDA graphs, API server and the startup probe. Phases are measured between
log events, so they always add up to the total. Give it two logs (weights from Hugging Face
vs weights mounted from a Cloud Storage bucket) and it compares them phase by phase.

    make logs                                            # saves data/startup_log.txt
    python experiments/01_cold_start_anatomy.py --log data/startup_log.txt
    python experiments/01_cold_start_anatomy.py --log hf.txt --compare gcs.txt
    python experiments/01_cold_start_anatomy.py --mock   # the checked-in sample
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "startup_log_sample.txt"

# (event, pattern in the log line, phase that ENDS at this event). Phases are built from the
# events sorted by time, so the order of this list does not matter (vLLM 0.33 logs the CUDA
# graphs before the KV-cache size; older versions did the opposite).
EVENTS = [
    ("instance_start", r"Starting new instance", None),
    ("first_log", r"non-default args|vLLM API server version", "container_start"),
    ("load_start", r"Loading model from scratch|Starting to load model", "process_init"),
    ("weights_downloaded", r"Time spent downloading weights", "download_weights"),
    ("weights_loaded", r"Model loading took", "load_weights_to_gpu"),
    ("compile_start", r"Starting torch\.compile", "model_setup"),
    ("compile_done", r"torch\.compile (?:takes|took)", "torch_compile"),
    ("graphs_done", r"Graph capturing finished", "cuda_graphs"),
    ("kv_cache_ready", r"GPU KV cache size", "kv_cache_profile"),
    ("engine_ready", r"init engine", "engine_init_rest"),
    ("server_up", r"Application startup complete|Starting vLLM API server", "api_server"),
    ("probe_ok", r"STARTUP .*probe succeeded", "startup_probe"),
]
# vLLM does not always log these: there is no download line when the weights come from a mount.
OPTIONAL = {"weights_downloaded", "compile_start"}
# Durations vLLM itself prints, to cross-check the timestamp arithmetic.
REPORTED = {
    "download_s": r"downloading weights.*?: ([\d.]+) seconds",
    "load_s": r"Loading weights took ([\d.]+) seconds",
    "model_loading_s": r"Model loading took .*? and ([\d.]+) seconds",
    "compile_s": r"torch\.compile (?:takes|took) ([\d.]+) s",
    "cuda_graphs_s": r"Graph capturing finished in ([\d.]+) secs",
    "engine_init_s": r"init engine .*? took ([\d.]+) seconds",
}
GROUPS = {
    "weights_s": ("download_weights", "load_weights_to_gpu"),
    "compile_s": ("torch_compile",),
    "cuda_graphs_s": ("cuda_graphs",),
}

_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\s+(.*)$")
_VLLM = re.compile(r"^\w+ (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def parse_line(line: str) -> tuple[float, str] | None:
    """gcloud 'value(timestamp,textPayload)' lines, or raw vLLM lines (second resolution)."""
    line = line.rstrip("\n")
    if not line.strip() or line.startswith("#"):
        return None
    m = _ISO.match(line)
    if m:
        frac = (m.group(2) or ".0")[:7]  # gcloud prints nanoseconds; Python takes micros
        ts = datetime.fromisoformat(m.group(1) + frac + "+00:00").timestamp()
        return ts, m.group(4)
    m = _VLLM.match(line)
    if m:
        mo, d, h, mi, s = (int(x) for x in m.groups())
        return datetime(2000, mo, d, h, mi, s).timestamp(), line
    return None


def anatomy(text: str) -> dict:
    lines = sorted(p for p in (parse_line(x) for x in text.splitlines()) if p)
    if not lines:
        raise ValueError("no timestamped lines found")
    found: dict[str, tuple[float, str]] = {}
    for ts, msg in lines:  # oldest first; keep the first occurrence of each event
        for event, pattern, _ in EVENTS:
            if event not in found and re.search(pattern, msg):
                found[event] = (ts, msg)
    t0 = found.get("instance_start", (lines[0][0], ""))[0]
    phases, prev_t = [], t0
    ending = {event: phase for event, _, phase in EVENTS[1:]}
    for event, (t, _) in sorted(((e, v) for e, v in found.items() if e in ending),
                                key=lambda item: item[1][0]):
        phases.append({"phase": ending[event], "ends_at_s": round(t - t0, 1),
                       "seconds": round(t - prev_t, 3)})
        prev_t = t
    total = prev_t - t0
    reported = {}
    for key, pattern in REPORTED.items():
        m = next((re.search(pattern, msg) for _, msg in lines if re.search(pattern, msg)), None)
        if m:
            reported[key] = float(m.group(1))
    by_phase = {p["phase"]: p["seconds"] for p in phases}
    summary = {k: round(sum(by_phase.get(p, 0.0) for p in v), 1) for k, v in GROUPS.items()}
    summary["rest_s"] = round(total - sum(summary.values()), 1)
    summary["total_s"] = round(total, 1)
    summary["total_min"] = round(total / 60, 2)
    slowest = max(phases, key=lambda p: p["seconds"])["phase"] if phases else None
    return {"phases": phases, "summary": summary, "reported_by_vllm": reported,
            "missing_events": [e for e, _, _ in EVENTS if e not in found and e not in OPTIONAL],
            "slowest_phase": slowest,
            "phases_sum_s": round(sum(p["seconds"] for p in phases), 3),
            "total_s": round(total, 3)}


def compare(a: dict, b: dict) -> dict:
    """a = baseline (e.g. Hugging Face), b = candidate (e.g. Cloud Storage mount)."""
    pa = {p["phase"]: p["seconds"] for p in a["phases"]}
    pb = {p["phase"]: p["seconds"] for p in b["phases"]}
    rows = [{"phase": ph, "a_s": round(pa.get(ph, 0.0), 1), "b_s": round(pb.get(ph, 0.0), 1),
             "saved_s": round(pa.get(ph, 0.0) - pb.get(ph, 0.0), 1)}
            for ph in dict.fromkeys([*pa, *pb])]
    return {"rows": rows, "a_total_s": a["summary"]["total_s"],
            "b_total_s": b["summary"]["total_s"],
            "saved_s": round(a["summary"]["total_s"] - b["summary"]["total_s"], 1),
            "speedup_x": round(a["summary"]["total_s"] / max(b["summary"]["total_s"], 1e-9), 2)}


def run(log_path: str | Path | None = None, compare_path: str | Path | None = None) -> dict:
    path = Path(log_path) if log_path else SAMPLE
    result = {"log": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
              "is_sample": path.resolve() == SAMPLE.resolve(),
              **anatomy(path.read_text())}
    if compare_path:
        other = anatomy(Path(compare_path).read_text())
        result["compare"] = {"log": str(compare_path), **compare(result, other)}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true", help="read the checked-in sample log")
    p.add_argument("--log", help="startup log exported with make logs")
    p.add_argument("--compare", help="second log to compare against --log")
    args = p.parse_args()
    r = run(None if args.mock else args.log, args.compare)
    print(f"\n== Cold start anatomy ({r['log']})")
    for ph in r["phases"]:
        bar = "#" * max(1, round(ph["seconds"] / 10))
        print(f"  {ph['phase']:22} {ph['seconds']:7.1f} s  {bar}")
    print(f"  {'total':22} {r['total_s']:7.1f} s  ({r['summary']['total_min']} min)")
    print(json.dumps({"summary": r["summary"], "reported_by_vllm": r["reported_by_vllm"]},
                     indent=2))
    if "compare" in r:
        print(json.dumps(r["compare"], indent=2))


if __name__ == "__main__":
    main()
