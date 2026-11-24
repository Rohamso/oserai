from __future__ import annotations

import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.config import AgentConfig
from agent.coordinator_client import CoordinatorClient
from agent.main import AgentState, ensure_registered, run_job
from common.hashing import sha256_hex
from coordinator import main
from coordinator.db import Database, SCHEMA_SQL
from coordinator.dispatch import dispatch_queued_jobs
from tests.test_agent_parallax import ANSWER, MODEL, response


@pytest.fixture
def api(db, config, monkeypatch):
    monkeypatch.setattr(main, "db", db, raising=False)
    monkeypatch.setattr(main, "config", config)
    # Without entering the lifespan: use the isolated test DB and manual
    # dispatch ticks, never the user's database or background dispatch loop.
    client = TestClient(main.app)
    yield client
    client.close()


@pytest.fixture
def parallax(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: response({"data": [{"id": MODEL}]}))
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response(ANSWER))


def register(api, backend, name=None):
    result = api.post("/nodes/register", json={
        "name": name or backend, "model_name": MODEL, "inference_backend": backend,
    })
    assert result.status_code == 200
    return result.json()["node_id"]


def test_parallax_batch_runs_through_agent_api_stats_and_ledger(api, db, config, tmp_path, parallax):
    agent_config = AgentConfig(
        node_name="test cluster", coordinator_url="http://testserver", model_name=MODEL,
        inference_backend="parallax", parallax_url="http://parallax",
        state_path=str(tmp_path / "state.json"),
    )
    bridge = CoordinatorClient("http://testserver")
    bridge._client.close()
    bridge._client = api
    ensure_registered(bridge, agent_config)
    created = api.post("/batches", json={
        "model_name": MODEL, "inference_backend": "parallax", "prompts": ["Capital of France?"],
    })
    assert created.status_code == 200
    batch_id = created.json()["batch_id"]
    assert db.get_batch(batch_id)["verification_pct"] == 0
    assert len(db.list_jobs_for_batch(batch_id)) == 1
    dispatch_queued_jobs(db, config)
    job = bridge.next_job()
    assert job["inference_backend"] == "parallax"
    state = AgentState()
    run_job(bridge, agent_config, job, state)

    progress = api.get(f"/batches/{batch_id}").json()
    assert progress["completed"] == 1
    assert progress["verified"] == 0
    assert progress["inference_backend"] == "parallax"
    result = api.get(f"/batches/{batch_id}/jobs").json()[0]
    assert result["output_text"] == "Paris."
    assert result["tokens_generated"] == 3
    assert result["duration_s"] >= 0
    assert db.get_job(job["job_id"])["output_sha256"] == sha256_hex("Paris.")
    ledger = db.list_ledger()
    assert len(ledger) == 1
    assert ledger[0]["payout"] == pytest.approx(3 * config.default_payout_rate)
    stats = api.get("/stats").json()
    assert stats["nodes"][0]["inference_backend"] == "parallax"
    assert stats["nodes"][0]["jobs_done"] == 1
    assert api.get("/nodes").json()[0]["inference_backend"] == "parallax"
    assert state.status == "idle"


def test_dispatch_never_crosses_backend_boundary_for_same_model(api, db, config):
    nodes = {backend: register(api, backend) for backend in ("ollama", "parallax")}
    for backend in nodes:
        batch = api.post("/batches", json={
            "model_name": MODEL, "prompts": ["hi"], "inference_backend": backend, "verification_pct": 0,
        }).json()
        job_id = db.list_jobs_for_batch(batch["batch_id"])[0]["id"]
        dispatch_queued_jobs(db, config)
        assert db.get_job(job_id)["assigned_node_id"] == nodes[backend]


def test_parallax_jobs_wait_instead_of_using_ollama_node(api, db, config):
    register(api, "ollama")
    batch = api.post("/batches", json={
        "model_name": MODEL, "prompts": ["hi"], "inference_backend": "parallax",
    }).json()
    assert dispatch_queued_jobs(db, config) == []
    assert api.get(f"/batches/{batch['batch_id']}").json()["queued"] == 1


def test_training_node_is_never_assigned_inference_work(api, db, config):
    db.register_node(
        name="trainer", token="token", city=None, country=None, hardware=None,
        model_name=MODEL, max_concurrent_jobs=1, measured_watts=None, grid_zone=None,
        role="training",
    )
    api.post("/batches", json={"model_name": MODEL, "prompts": ["hi"], "verification_pct": 0})
    assert dispatch_queued_jobs(db, config) == []


def test_parallax_rejects_exact_verification_before_creating_batch(api, db):
    result = api.post("/batches", json={
        "model_name": MODEL, "prompts": ["hi"], "inference_backend": "parallax", "verification_pct": 0.2,
    })
    assert result.status_code == 400
    assert "verification_pct=0" in result.json()["detail"]
    assert db.list_batches() == []


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_verification_probability_is_bounded(api, value):
    assert api.post("/batches", json={"model_name": MODEL, "prompts": ["hi"], "verification_pct": value}).status_code == 422


def test_legacy_batches_default_to_ollama_with_existing_verification(api, db, monkeypatch):
    monkeypatch.setattr(main.random, "random", lambda: 0)
    batch = api.post("/batches", json={"model_name": MODEL, "prompts": ["hi"]}).json()
    assert db.get_batch(batch["batch_id"])["inference_backend"] == "ollama"
    assert len(db.list_jobs_for_batch(batch["batch_id"])) == 2
    # Customer results contain only the primary prompt, never duplicate work.
    assert len(api.get(f"/batches/{batch['batch_id']}/jobs").json()) == 1


def test_backend_failure_requeues_then_succeeds_without_duplicate_payout(api, db, config, tmp_path, parallax, monkeypatch):
    agent_config = AgentConfig(
        node_name="cluster", coordinator_url="http://testserver", model_name=MODEL,
        inference_backend="parallax", parallax_url="http://parallax", state_path=str(tmp_path / "state.json"),
    )
    bridge = CoordinatorClient("http://testserver")
    bridge._client.close()
    bridge._client = api
    ensure_registered(bridge, agent_config)
    batch_id = api.post("/batches", json={
        "model_name": MODEL, "prompts": ["hi"], "inference_backend": "parallax",
    }).json()["batch_id"]
    dispatch_queued_jobs(db, config)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response({"error": "not ready"}, 503))
    state = AgentState()
    run_job(bridge, agent_config, bridge.next_job(), state)
    assert api.get(f"/batches/{batch_id}").json()["queued"] == 1
    assert db.list_ledger() == []
    assert state.status == "idle"
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response(ANSWER))
    dispatch_queued_jobs(db, config)
    run_job(bridge, agent_config, bridge.next_job(), state)
    assert api.get(f"/batches/{batch_id}").json()["completed"] == 1
    assert api.get(f"/batches/{batch_id}/jobs").json()[0]["error"] is None
    assert len(db.list_ledger()) == 1


def test_replayed_success_cannot_double_count_usage_or_free_busy_bridge(api, db, config):
    registered = api.post("/nodes/register", json={
        "name": "cluster", "model_name": MODEL, "inference_backend": "parallax",
    }).json()
    node_id = registered["node_id"]
    headers = {"Authorization": f"Bearer {registered['token']}"}
    batch_id = api.post("/batches", json={
        "model_name": MODEL, "prompts": ["first", "second"], "inference_backend": "parallax",
    }).json()["batch_id"]
    dispatch_queued_jobs(db, config)
    first = api.get("/jobs/next", params={"node_id": node_id}, headers=headers).json()
    payload = {"node_id": node_id, "success": True, "output_text": "answer", "tokens_generated": 5, "duration_s": 1.0}
    assert api.post(f"/jobs/{first['job_id']}/result", json=payload, headers=headers).status_code == 200
    dispatch_queued_jobs(db, config)
    api.get("/jobs/next", params={"node_id": node_id}, headers=headers)
    assert api.post(f"/jobs/{first['job_id']}/result", json=payload, headers=headers).status_code == 200
    assert len(db.list_ledger()) == 1
    assert db.get_node(node_id)["tokens_total"] == 5
    assert db.get_node(node_id)["status"] == "busy"
    assert api.get(f"/batches/{batch_id}").json()["running"] == 1


def test_switching_backends_reregisters_and_retires_old_identity(api, db, tmp_path):
    agent_config = AgentConfig(
        node_name="bridge", coordinator_url="http://testserver", model_name=MODEL,
        state_path=str(tmp_path / "state.json"),
    )
    bridge = CoordinatorClient("http://testserver")
    bridge._client.close()
    bridge._client = api
    ensure_registered(bridge, agent_config)
    original = bridge.node_id
    ensure_registered(bridge, agent_config)
    assert bridge.node_id == original
    agent_config.inference_backend = "parallax"
    ensure_registered(bridge, agent_config)
    assert bridge.node_id != original
    assert db.get_node(original)["status"] == "offline"
    assert db.get_node(bridge.node_id)["inference_backend"] == "parallax"


def test_old_database_is_migrated_and_reopens_without_losing_rows(tmp_path):
    path = str(tmp_path / "legacy.db")
    legacy_schema = "\n".join(line for line in SCHEMA_SQL.splitlines() if "inference_backend" not in line)
    connection = sqlite3.connect(path)
    connection.executescript(legacy_schema)
    connection.execute("INSERT INTO nodes (id, name, token, model_name, created_at) VALUES ('old', 'old node', 'token', 'model', 1)")
    connection.execute("INSERT INTO batches VALUES ('batch', 'model', 100, 300, 0, 0.1, 1, 1)")
    connection.commit()
    connection.close()
    for _ in range(2):
        migrated = Database(path)
        assert migrated.get_node("old")["inference_backend"] == "ollama"
        assert migrated.get_batch("batch")["inference_backend"] == "ollama"
        assert migrated.get_node("old")["token"] == "token"
        migrated.close()


def test_yaml_and_environment_select_backend_and_endpoint(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("node_name: cluster\ncoordinator_url: http://localhost:8000\nmodel_name: test\n")
    monkeypatch.setenv("AGENT_INFERENCE_BACKEND", "parallax")
    monkeypatch.setenv("AGENT_PARALLAX_URL", "http://parallax:3001")
    loaded = AgentConfig.load(str(path))
    assert loaded.inference_backend == "parallax"
    assert loaded.parallax_url == "http://parallax:3001"
    monkeypatch.setenv("AGENT_INFERENCE_BACKEND", "typo")
    with pytest.raises(ValueError, match="inference_backend"):
        AgentConfig.load(str(path))
