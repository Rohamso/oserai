"""Coordinator FastAPI app: node registry, batch/job dispatch, verification,
simulated ledger, stats, and the dashboard."""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse

from common.logging_setup import log_event, setup_logging
from coordinator.config import CoordinatorConfig
from coordinator.db import Database
from coordinator.dispatch import handle_job_result, run_dispatch_cycle
from coordinator.models import (
    BatchCreateRequest,
    BatchCreateResponse,
    BatchProgress,
    HeartbeatRequest,
    JobNextResponse,
    JobResultRequest,
    NodeOut,
    NodeRegisterRequest,
    NodeRegisterResponse,
    NodeStats,
    StatsResponse,
)
from coordinator.stats import compute_batch_progress, compute_node_stats

STATIC_DIR = Path(__file__).parent / "static"

config = CoordinatorConfig.load()
logger = setup_logging("coordinator", level=getattr(logging, config.log_level, logging.INFO))

db: Database  # assigned in lifespan startup


async def _dispatch_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(run_dispatch_cycle, db, config)
        except Exception:
            logger.exception("dispatch cycle failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=config.dispatch_interval_s)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = Database(config.db_path)
    stop_event = asyncio.Event()
    task = asyncio.create_task(_dispatch_loop(stop_event))
    log_event(logger, "coordinator_started", db_path=config.db_path, port=config.port)
    yield
    stop_event.set()
    task.cancel()
    db.close()


app = FastAPI(title="Distributed Inference Coordinator", lifespan=lifespan)


def _auth_node(node_id: str, authorization: Optional[str]) -> dict:
    node = db.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="unknown node")
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or token != node["token"]:
        raise HTTPException(status_code=401, detail="invalid or missing token")
    return node


def _node_to_out(node: dict) -> NodeOut:
    tps = node["tokens_total"] / node["duration_total"] if node["duration_total"] else 0.0
    return NodeOut(**{**node, "tokens_per_sec_avg": tps})


# ---------------------------------------------------------------------- #
# node registry
# ---------------------------------------------------------------------- #

@app.post("/nodes/register", response_model=NodeRegisterResponse)
def register_node(req: NodeRegisterRequest) -> NodeRegisterResponse:
    token = secrets.token_hex(16)
    node_id = db.register_node(
        name=req.name, token=token, city=req.city, country=req.country,
        hardware=req.hardware, model_name=req.model_name,
        max_concurrent_jobs=req.max_concurrent_jobs,
        measured_watts=req.measured_watts, grid_zone=req.grid_zone,
    )
    log_event(logger, "node_registered", node_id=node_id, name=req.name, model_name=req.model_name)
    return NodeRegisterResponse(node_id=node_id, token=token)


@app.post("/nodes/{node_id}/heartbeat")
def heartbeat(node_id: str, req: HeartbeatRequest, authorization: Optional[str] = Header(None)) -> dict:
    _auth_node(node_id, authorization)
    db.update_heartbeat(node_id, req.status, req.cpu_pct, req.ram_free_mb)
    return {"ok": True}


@app.post("/nodes/{node_id}/deregister")
def deregister(node_id: str, authorization: Optional[str] = Header(None)) -> dict:
    _auth_node(node_id, authorization)
    db.deregister_node(node_id)
    log_event(logger, "node_deregistered", node_id=node_id)
    return {"ok": True}


@app.get("/nodes", response_model=list[NodeOut])
def list_nodes() -> list[NodeOut]:
    return [_node_to_out(n) for n in db.list_nodes()]


# ---------------------------------------------------------------------- #
# batches / jobs
# ---------------------------------------------------------------------- #

@app.post("/batches", response_model=BatchCreateResponse)
def create_batch(req: BatchCreateRequest) -> BatchCreateResponse:
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts must not be empty")
    verification_pct = req.verification_pct if req.verification_pct is not None else config.default_verification_pct

    batch_id = db.create_batch(
        model_name=req.model_name, max_tokens=req.max_tokens, timeout_s=req.timeout_s,
        verification_pct=verification_pct, payout_rate=config.default_payout_rate,
        total_jobs=len(req.prompts),
    )
    for prompt in req.prompts:
        if random.random() < verification_pct:
            group_id = str(uuid.uuid4())
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s, verification_group_id=group_id, is_duplicate=False)
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s, verification_group_id=group_id, is_duplicate=True)
        else:
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s)

    log_event(logger, "batch_created", batch_id=batch_id, total_jobs=len(req.prompts), model_name=req.model_name)
    return BatchCreateResponse(batch_id=batch_id, total_jobs=len(req.prompts))


@app.get("/batches/{batch_id}", response_model=BatchProgress)
def get_batch(batch_id: str) -> BatchProgress:
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    return BatchProgress(**compute_batch_progress(db, batch))


@app.get("/jobs/next")
def next_job(node_id: str, authorization: Optional[str] = Header(None)):
    _auth_node(node_id, authorization)
    job = db.get_dispatched_job_for_node(node_id)
    if job is None:
        return Response(status_code=204)
    db.mark_running(job["id"])
    batch = db.get_batch(job["batch_id"])
    return JobNextResponse(
        job_id=job["id"], prompt=job["prompt"], max_tokens=job["max_tokens"],
        timeout_s=job["timeout_s"], model_name=batch["model_name"],
    )


@app.post("/jobs/{job_id}/result")
def post_result(job_id: str, req: JobResultRequest, authorization: Optional[str] = Header(None)) -> dict:
    _auth_node(req.node_id, authorization)
    try:
        job = handle_job_result(
            db, config, job_id, req.node_id, req.success,
            req.output_text, req.output_sha256, req.tokens_generated,
            req.duration_s, req.tokens_per_sec, req.error,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "status": job["status"]}


# ---------------------------------------------------------------------- #
# stats + dashboard
# ---------------------------------------------------------------------- #

@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    nodes = [NodeStats(**compute_node_stats(n, config.heartbeat_timeout_s)) for n in db.list_nodes()]
    batches = [BatchProgress(**compute_batch_progress(db, b)) for b in db.list_batches()]
    aggregate_tps = sum(n.tokens_per_sec_avg for n in nodes if n.status != "offline")
    return StatsResponse(
        nodes=nodes, batches=batches,
        aggregate_tokens_per_sec=aggregate_tps,
        cumulative_payout=db.total_payouts(),
    )


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (STATIC_DIR / "dashboard.html").read_text()
