"""Just enough of the Prometheus text format to read vLLM's /metrics without extra packages.

vLLM renames metrics between versions (for example `gpu_cache_usage_perc` became
`kv_cache_usage_perc`), so lookups go by name fragments instead of exact names.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def parse(text: str) -> list[Sample]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        samples.append(Sample(m.group(1), dict(_LABEL.findall(m.group(3) or "")), value))
    return samples


def _short(name: str) -> str:
    for prefix in ("vllm:", "vllm_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def find_name(samples: list[Sample], fragments: tuple[str, ...], suffix: str = "") -> str | None:
    """First metric name (without labels) that contains any fragment and ends with suffix."""
    for frag in fragments:
        for s in samples:
            short = _short(s.name)
            if frag in short and short.endswith(suffix):
                return s.name
    return None


def gauge(samples: list[Sample], fragments: tuple[str, ...], suffix: str = "") -> float | None:
    """Sum of every series of the first matching metric (sums across engines/labels)."""
    name = find_name(samples, fragments, suffix)
    if name is None:
        return None
    return sum(s.value for s in samples if s.name == name)


def buckets(samples: list[Sample], fragments: tuple[str, ...]) -> list[tuple[float, float]]:
    """Cumulative (upper_bound, count) pairs of a histogram, summed across label sets."""
    name = find_name(samples, fragments, "_bucket")
    if name is None:
        return []
    acc: dict[float, float] = {}
    for s in samples:
        if s.name == name and "le" in s.labels:
            le = math.inf if s.labels["le"] in ("+Inf", "Inf") else float(s.labels["le"])
            acc[le] = acc.get(le, 0.0) + s.value
    return sorted(acc.items())


def diff_buckets(after: list[tuple[float, float]],
                 before: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Histogram of only what happened between two scrapes."""
    prev = dict(before)
    return [(le, c - prev.get(le, 0.0)) for le, c in after]


def quantile(q: float, cumulative: list[tuple[float, float]]) -> float | None:
    """Same linear interpolation as PromQL's histogram_quantile."""
    if not cumulative or cumulative[-1][1] <= 0:
        return None
    total = cumulative[-1][1]
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in cumulative:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le
