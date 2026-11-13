"""Runs every experiment against the built-in mock service (no GPU needed)."""

import importlib.util
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from vllm_lab import mock_server, prom

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "startup_log_sample.txt"


@pytest.fixture(scope="module")
def base_url():
    server, url = mock_server.start()
    yield url
    server.shutdown()


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_01_phases_add_up_to_the_measured_cold_start():
    r = load("01_cold_start_anatomy").run()
    assert r["is_sample"] and not r["missing_events"]
    assert abs(r["phases_sum_s"] - r["total_s"]) < 1e-6
    assert r["slowest_phase"] == "download_weights"
    s = r["summary"]
    assert 500 <= s["total_s"] <= 520 and s["compile_s"] == 55.0 and s["cuda_graphs_s"] > 9
    assert r["reported_by_vllm"]["download_s"] == 262.0


def test_01_compare_faster_weights(tmp_path):
    """Shorten only the download (as with weights in a bucket) and compare the two logs."""
    mod = load("01_cold_start_anatomy")
    out = []
    for line in SAMPLE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        ts, msg = line.split("\t", 1)
        t = datetime.fromisoformat(ts)
        if t >= datetime.fromisoformat("2026-11-10T15:06:32+00:00"):
            t -= timedelta(seconds=200)
        out.append(f"{t.isoformat().replace('+00:00', 'Z')}\t{msg}")
    fast = tmp_path / "gcs.txt"
    fast.write_text("\n".join(out) + "\n")
    r = mod.run(SAMPLE, fast)
    rows = {row["phase"]: row for row in r["compare"]["rows"]}
    assert rows["download_weights"]["saved_s"] == 200.0
    assert r["compare"]["saved_s"] == 200.0 and r["compare"]["speedup_x"] > 1.6


def test_02_ready_and_probe_budget(base_url):
    r = load("02_health_and_readiness").run(base_url)
    assert r["health"]["status"] == 200 and r["readiness"]["ready"]
    assert r["models"]["id"] == "gemma" and r["version"]["vllm"] == "mock"
    assert r["startup_probe"]["budget_s"] == 1500 and r["startup_probe"]["fits"]


def test_03_metrics_parsed(base_url):
    r = load("03_metrics_dashboard").run(base_url, users=4, max_tokens=16)
    assert r["parsed_all"]
    assert r["generation_tokens_in_burst"] == r["expected_generation_tokens"] == 64
    assert r["latency"]["e2e"]["p95_s"] >= r["latency"]["ttft"]["p95_s"]


def test_prom_tolerates_renamed_metrics():
    text = (
        'vllm:gpu_cache_usage_perc{model_name="g"} 0.25\n'
        'vllm:time_to_first_token_seconds_bucket{le="0.1"} 5\n'
        'vllm:time_to_first_token_seconds_bucket{le="0.5"} 10\n'
        'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10\n'
    )
    s = prom.parse(text)
    assert prom.gauge(s, ("kv_cache_usage_perc", "gpu_cache_usage_perc")) == 0.25
    b = prom.buckets(s, ("time_to_first_token",))
    assert prom.quantile(0.5, b) == pytest.approx(0.1)
    assert prom.quantile(0.75, b) == pytest.approx(0.3)


def test_04_second_instance_lowers_p95(base_url):
    r = load("04_autoscaling_two_instances").run(base_url, users=24, duration_s=4.0)
    assert r["instances_seen"] == 2 and r["errors"] == 0
    assert r["p95_after_s"] < r["p95_before_s"]


def test_05_breakeven(base_url):
    r = load("05_cost_vs_api").run(base_url, out_tokens=64)
    one, many = r["rows"]
    assert many["usd_per_million_output_tokens"] < one["usd_per_million_output_tokens"]
    be = r["current"]["breakeven_tokens_per_day"]
    assert be > 0 and math.isfinite(be)
    later = r["by_api_period"]["desde 2027-01-01"]["breakeven_tokens_per_day"]
    assert later < be  # a pricier API breaks even sooner
    assert r["current"]["idle"]["min_instances_1_usd_per_month"] == pytest.approx(1037.26)
    assert r["current"]["idle"]["wake_assumes_s"]["idle_before_scale_down"] == r["idle_before_scale_down"]["seconds"]


def test_05_idle_time_comes_from_the_lifecycle_log():
    mod = load("05_cost_vs_api")
    idle = mod.idle_before_scale_down()
    # vllm-gemma-00010-bmh: probe passed 18:11:01.346, shutting down 18:21:15.302 (10.2 min)
    assert idle["seconds"] == pytest.approx(614.0, abs=0.1)
    assert mod.wake_usd(1.4209, 580.351, idle["seconds"]) == pytest.approx(0.471, abs=0.001)


def test_02_probe_budget_uses_the_cold_start_it_is_given():
    probe = load("02_health_and_readiness").probe_budget(cold_start_s=580.351)
    assert probe["measured_cold_start_s"] == 580.351 and probe["headroom_x"] == pytest.approx(2.58)


def test_01_reads_a_real_vllm_033_log_and_matches_what_vllm_reports():
    r = load("01_cold_start_anatomy").run(ROOT / "data" / "startup_log_v033_excerpt.txt")
    by_phase = {p["phase"]: p["seconds"] for p in r["phases"]}
    assert abs(by_phase["load_weights_to_gpu"] - r["reported_by_vllm"]["model_loading_s"]) < 2
    assert abs(by_phase["torch_compile"] - r["reported_by_vllm"]["compile_s"]) < 1
    assert not r["missing_events"] and r["phases_sum_s"] == pytest.approx(r["total_s"], abs=1e-3)
    assert r["slowest_phase"] == "load_weights_to_gpu"
