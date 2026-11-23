"""A fake of a vLLM service on Cloud Run, for running everything without a GPU.

It speaks the HTTP that matters in production (/health, /version, /v1/models, /metrics and
/v1/chat/completions) and simulates the part a single server cannot show: Cloud Run's
autoscaler. Each instance accepts up to `concurrency` requests at a time; the rest wait in
line. When a request has to wait and `max_instances` allows it, a second instance starts and
becomes ready after `scale_delay_s` (the cold start, compressed from minutes to a second).

Every response carries an `x-instance-id` header so the load generator can tell instances
apart; stock vLLM behind Cloud Run sends no such header. Each instance also exposes its own
`process_start_time_seconds` in /metrics, as a real one does. The numbers are made up on
purpose; only the live judge (`make verify-live`) measures a real GPU.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_FIXED_S = 0.010
PREFILL_PER_TOKEN_S = 0.00004
DECODE_PER_TOKEN_S = 0.012
BATCH_SLOWDOWN = 0.05  # each extra running request slows a decode step by 5 %
NATURAL_LENGTH = 40
BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 30.0,
           60.0)

_SENTENCE = (
    "En producción un servidor de inferencia necesita tres cosas: arrancar a tiempo, avisar "
    "cuando está listo y contar lo que hace para saber cuánto cuesta cada token."
)
WORDS = _SENTENCE.split(" ")


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def fake_tokens(prompt: str, n: int) -> list[str]:
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(WORDS)
    return [WORDS[(start + i) % len(WORDS)] + " " for i in range(n)]


class Histogram:
    def __init__(self) -> None:
        self.counts = [0] * len(BUCKETS)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for i, edge in enumerate(BUCKETS):
            if value <= edge:
                self.counts[i] += 1

    def lines(self, name: str, labels: str) -> list[str]:
        out = [f"# TYPE {name} histogram"]
        for edge, c in zip(BUCKETS, self.counts, strict=True):
            out.append(f'{name}_bucket{{{labels},le="{edge}"}} {c}')
        out.append(f'{name}_bucket{{{labels},le="+Inf"}} {self.total}')
        out.append(f"{name}_count{{{labels}}} {self.total}")
        out.append(f"{name}_sum{{{labels}}} {self.sum:.6f}")
        return out


class Instance:
    def __init__(self, idx: int, ready_at: float, concurrency: int) -> None:
        self.id = f"mock-instance-{idx}"
        self.ready_at = ready_at
        self.started_wall = time.time() + (ready_at - time.monotonic())
        self.slots = threading.Semaphore(concurrency)
        self.running = 0
        self.waiting = 0
        self.ttft = Histogram()
        self.e2e = Histogram()
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.success = 0

    def ready(self) -> bool:
        return time.monotonic() >= self.ready_at


class Cluster:
    """The autoscaler: picks an instance per request and starts a new one under pressure."""

    def __init__(self, concurrency: int, max_instances: int, scale_delay_s: float) -> None:
        self.concurrency = concurrency
        self.max_instances = max_instances
        self.scale_delay_s = scale_delay_s
        self.lock = threading.Lock()
        self.instances = [Instance(0, time.monotonic(), concurrency)]
        self.rr = 0

    def pick(self) -> Instance:
        with self.lock:
            ready = [i for i in self.instances if i.ready()]
            inst = min(ready, key=lambda i: i.running + i.waiting)
            if (inst.running + inst.waiting >= self.concurrency
                    and len(self.instances) < self.max_instances):
                ready_at = time.monotonic() + self.scale_delay_s
                self.instances.append(Instance(len(self.instances), ready_at, self.concurrency))
            return inst

    def any_ready(self) -> Instance:
        with self.lock:
            ready = [i for i in self.instances if i.ready()]
            self.rr += 1
            return ready[self.rr % len(ready)]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _MockServer

    def log_message(self, *args):  # keep the terminal clean
        pass

    def _send(self, code: int, data: bytes, ctype: str, instance: Instance) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-instance-id", instance.id)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code: int, body: dict, instance: Instance) -> None:
        self._send(code, json.dumps(body).encode(), "application/json", instance)

    def do_GET(self):
        inst = self.server.cluster.any_ready()
        if self.path == "/health":
            self._send(200, b"", "text/plain", inst)
        elif self.path == "/version":
            self._json(200, {"version": "mock"}, inst)
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "gemma", "object": "model", "owned_by": "vllm", "max_model_len": 8192,
                 "root": "/models/google/gemma-4-E4B-it"}]}, inst)
        elif self.path == "/metrics":
            self._send(200, metrics_text(inst).encode(), "text/plain; version=0.0.4", inst)
        else:
            self._json(404, {"error": "not found"}, inst)

    def do_POST(self):
        cluster = self.server.cluster
        inst = cluster.pick()
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"}, inst)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "".join(m["content"] for m in body["messages"])
        max_tokens = int(body.get("max_tokens") or 256)
        n = max_tokens if body.get("ignore_eos") else min(max_tokens, NATURAL_LENGTH)
        finish = "length" if n == max_tokens else "stop"
        ptoks = count_tokens(prompt)
        tokens = fake_tokens(prompt, n)
        usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n}

        t0 = time.monotonic()
        with cluster.lock:
            inst.waiting += 1
        inst.slots.acquire()
        with cluster.lock:
            inst.waiting -= 1
            inst.running += 1
        try:
            time.sleep(PREFILL_FIXED_S + PREFILL_PER_TOKEN_S * ptoks)
            inst.ttft.observe(time.monotonic() - t0)
            if body.get("stream"):
                self._stream(tokens, finish, usage, body, inst)
            else:
                for _ in tokens[1:]:
                    time.sleep(self._step(inst))
                self._json(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "model": "gemma",
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": "".join(tokens)}}],
                    "usage": usage}, inst)
        finally:
            with cluster.lock:
                inst.running -= 1
                inst.e2e.observe(time.monotonic() - t0)
                inst.prompt_tokens += ptoks
                inst.generation_tokens += n
                inst.success += 1
            inst.slots.release()

    def _stream(self, tokens, finish, usage, body, inst) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("x-instance-id", inst.id)
        self.end_headers()
        for i, tok in enumerate(tokens):
            if i:
                time.sleep(self._step(inst))
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                         "model": "gemma", "choices": [{"index": 0, "delta": {"content": tok},
                                                        "finish_reason": None}]})
        self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": "gemma",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                         "model": "gemma", "choices": [], "usage": usage})
        self._raw(b"data: [DONE]\n\n")
        self._raw(b"")

    @staticmethod
    def _step(inst: Instance) -> float:
        return DECODE_PER_TOKEN_S * (1 + BATCH_SLOWDOWN * max(inst.running - 1, 0))

    def _chunk(self, obj: dict) -> None:
        self._raw(f"data: {json.dumps(obj)}\n\n".encode())

    def _raw(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


def metrics_text(inst: Instance) -> str:
    """Prometheus text with the names vLLM uses (prefix `vllm:`, label model_name)."""
    lab = 'engine="0",model_name="gemma"'
    lines = [
        "# TYPE process_start_time_seconds gauge",
        f"process_start_time_seconds {inst.started_wall:.2f}",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{lab}}} {inst.running}.0",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{lab}}} {inst.waiting}.0",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{{{lab}}} {min(1.0, 0.03 * inst.running):.4f}",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total{{{lab}}} {inst.prompt_tokens}.0",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{{{lab}}} {inst.generation_tokens}.0",
        "# TYPE vllm:request_success_total counter",
        f'vllm:request_success_total{{{lab},finished_reason="stop"}} {inst.success}.0',
    ]
    lines += inst.ttft.lines("vllm:time_to_first_token_seconds", lab)
    lines += inst.e2e.lines("vllm:e2e_request_latency_seconds", lab)
    return "\n".join(lines) + "\n"


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    cluster: Cluster

    def handle_error(self, request, client_address):  # clients closing keep-alive sockets
        pass


def start(port: int = 0, concurrency: int = 8, max_instances: int = 2,
          scale_delay_s: float = 1.5) -> tuple[ThreadingHTTPServer, str]:
    """Starts the mock in a background thread; returns the server and its /v1 base URL."""
    server = _MockServer(("127.0.0.1", port), Handler)
    server.cluster = Cluster(concurrency, max_instances, scale_delay_s)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(8000)
    print(f"Mock vLLM service at {url} (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
