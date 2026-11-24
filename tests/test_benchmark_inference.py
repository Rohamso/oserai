from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.config import AgentConfig
from common.inference import InferenceError, InferenceResult
from scripts import benchmark_inference


def test_benchmark_counts_failures_in_overall_throughput(monkeypatch):
    clock = iter([0, 1, 3, 4, 9, 10])
    monkeypatch.setattr(benchmark_inference, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    def inference(config, model, prompt, max_tokens, timeout_s):
        if prompt == "bad":
            raise InferenceError("worker unavailable")
        return InferenceResult("answer", 20, 2.0, 10.0)
    monkeypatch.setattr(benchmark_inference, "run_inference", inference)
    config = AgentConfig("cluster", "http://localhost:8000", "model", inference_backend="parallax")

    report = benchmark_inference.benchmark(config, ["good", "bad"], 100, 30)

    assert report["succeeded"] == report["failed"] == 1
    assert report["success_rate"] == 0.5
    assert report["median_latency_s"] == 2
    assert report["end_to_end_tokens_per_sec"] == 2
    assert report["results"][1]["error"] == "worker unavailable"


def test_failed_benchmark_still_writes_answers_and_failure_report(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("node_name: cluster\ncoordinator_url: http://localhost:8000\nmodel_name: model\ninference_backend: parallax\n")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    output = tmp_path / "reports" / "benchmark.json"
    def failed_inference(*args):
        raise InferenceError("cluster offline")
    monkeypatch.setattr(benchmark_inference, "run_inference", failed_inference)
    monkeypatch.setattr("sys.argv", ["benchmark_inference", str(prompts), "--config", str(config), "--output", str(output)])

    with pytest.raises(SystemExit) as error:
        benchmark_inference.main()

    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report["failed"] == 1
    assert report["median_latency_s"] is None
    assert report["results"][0]["error"] == "cluster offline"
