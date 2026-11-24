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

from fastapi import FastAPI, Header, HTTPException, Request, Response
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
    JobOut,
    JobResultRequest,
    NodeOut,
    NodeRegisterRequest,
    NodeRegisterResponse,
    NodeStats,
    StatsResponse,
)
from coordinator.stats import compute_batch_progress, compute_node_stats

try:
    import torch

    from coordinator.models import (
        GenerateRequest,
        GenerateResponse,
        TrainingDeltaSubmitResponse,
        TrainingNodeRegisterRequest,
        TrainingRoundOut,
        TrainingRunCreateRequest,
        TrainingRunCreateResponse,
        TrainingRunOut,
    )
    from coordinator.training_dispatch import run_training_cycle, start_run, submit_delta
    from training.checkpoint_io import bytes_to_state_dict, state_dict_to_bytes
    from training.config import GPTConfig
    from training.dataset import CHAT_DATASETS, CHAT_TEMPLATE, DATASET_NAMES, VOCAB_SIZE, eot_token_id, get_encoder
    from training.model import GPT
    from training.lora import (
        DEFAULT_BASE_MODEL_ID,
        DEFAULT_BLOCK_SIZE as DEFAULT_LORA_BLOCK_SIZE,
        DEFAULT_LORA_ALPHA,
        DEFAULT_LORA_DROPOUT,
        DEFAULT_LORA_R,
        DEFAULT_LORA_TARGET_MODULES,
        LoraRunConfig,
        generate_chat_reply,
        get_adapter_state_dict,
        load_adapter_state_dict,
        load_frozen_base,
        load_tokenizer,
        temporary_lora_adapter,
    )

    LORA_BACKEND_PREFIX = "lora_"
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

STATIC_DIR = Path(__file__).parent / "static"

config = CoordinatorConfig.load()
logger = setup_logging("coordinator", level=getattr(logging, config.log_level, logging.INFO))
if not TRAINING_AVAILABLE:
    logger.warning("torch not installed -- /training/* routes disabled (see requirements-training.txt)")

db: Database  # assigned in lifespan startup


async def _dispatch_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(run_dispatch_cycle, db, config)
            if TRAINING_AVAILABLE:
                await asyncio.to_thread(run_training_cycle, db)
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
        inference_backend=req.inference_backend,
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
    if req.inference_backend == "parallax":
        if req.verification_pct not in (None, 0):
            raise HTTPException(
                status_code=400,
                detail="Parallax batches require verification_pct=0: exact-output verification is not supported for shared clusters",
            )
        verification_pct = 0.0
    else:
        verification_pct = req.verification_pct if req.verification_pct is not None else config.default_verification_pct

    batch_id = db.create_batch(
        model_name=req.model_name, max_tokens=req.max_tokens, timeout_s=req.timeout_s,
        verification_pct=verification_pct, payout_rate=config.default_payout_rate,
        total_jobs=len(req.prompts),
        inference_backend=req.inference_backend,
    )
    for prompt in req.prompts:
        if random.random() < verification_pct:
            group_id = str(uuid.uuid4())
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s, verification_group_id=group_id, is_duplicate=False)
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s, verification_group_id=group_id, is_duplicate=True)
        else:
            db.create_job(batch_id, prompt, req.max_tokens, req.timeout_s)

    log_event(logger, "batch_created", batch_id=batch_id, total_jobs=len(req.prompts), model_name=req.model_name, inference_backend=req.inference_backend)
    return BatchCreateResponse(batch_id=batch_id, total_jobs=len(req.prompts))


@app.get("/batches/{batch_id}", response_model=BatchProgress)
def get_batch(batch_id: str) -> BatchProgress:
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    return BatchProgress(**compute_batch_progress(db, batch))


@app.get("/batches/{batch_id}/jobs", response_model=list[JobOut])
def get_batch_jobs(batch_id: str) -> list[JobOut]:
    if db.get_batch(batch_id) is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    return [JobOut(**job) for job in db.list_jobs_for_batch(batch_id) if not job["is_duplicate"]]


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
        inference_backend=batch["inference_backend"],
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


# ---------------------------------------------------------------------- #
# training (DiLoCo) -- only registered if torch is installed
# ---------------------------------------------------------------------- #

if TRAINING_AVAILABLE:

    def _training_run_to_out(run: dict) -> TrainingRunOut:
        return TrainingRunOut(**run)

    @app.post("/training/nodes/register", response_model=NodeRegisterResponse)
    def register_training_node(req: TrainingNodeRegisterRequest) -> NodeRegisterResponse:
        token = secrets.token_hex(16)
        node_id = db.register_node(
            name=req.name, token=token, city=req.city, country=req.country,
            hardware=req.hardware, model_name="training", max_concurrent_jobs=1,
            measured_watts=None, grid_zone=None, role="training",
        )
        log_event(logger, "training_node_registered", node_id=node_id, name=req.name)
        return NodeRegisterResponse(node_id=node_id, token=token)

    @app.post("/training/runs", response_model=TrainingRunCreateResponse)
    def create_training_run(req: TrainingRunCreateRequest) -> TrainingRunCreateResponse:
        if req.dataset_name not in DATASET_NAMES:
            raise HTTPException(status_code=400, detail=f"unknown dataset_name, expected one of {sorted(DATASET_NAMES)}")

        if req.model_backend.startswith(LORA_BACKEND_PREFIX):
            if req.init_from_run_id:
                raise HTTPException(
                    status_code=400,
                    detail="init_from_run_id isn't supported for lora_* backends -- every LoRA run already "
                           "starts from the same pretrained base_model_id, there's no separate pretrain stage",
                )

            base_model_id = req.base_model_id or DEFAULT_BASE_MODEL_ID
            block_size = req.block_size or DEFAULT_LORA_BLOCK_SIZE
            lora_config = LoraRunConfig(
                base_model_id=base_model_id,
                lora_r=req.lora_r or DEFAULT_LORA_R,
                lora_alpha=req.lora_alpha or DEFAULT_LORA_ALPHA,
                lora_dropout=req.lora_dropout if req.lora_dropout is not None else DEFAULT_LORA_DROPOUT,
                lora_target_modules=tuple(req.lora_target_modules) if req.lora_target_modules else DEFAULT_LORA_TARGET_MODULES,
            )
            torch.manual_seed(1337)  # the coordinator's one canonical round-0 adapter every node downloads
            base = load_frozen_base(base_model_id, device="cpu", dtype=torch.float16)
            tokenizer = load_tokenizer(base_model_id)
            with temporary_lora_adapter(base, lora_config) as peft_model:
                initial_state_bytes = state_dict_to_bytes(get_adapter_state_dict(peft_model))

            run_id = db.create_training_run(
                name=req.name, vocab_size=tokenizer.vocab_size, dataset_name=req.dataset_name,
                inner_lr=req.inner_lr, inner_steps_h=req.inner_steps_h, inner_batch_size=req.inner_batch_size,
                outer_lr=req.outer_lr, outer_momentum=req.outer_momentum, max_rounds=req.max_rounds,
                round_timeout_s=req.round_timeout_s, quorum=req.quorum, block_size=block_size,
                model_backend=req.model_backend, base_model_id=base_model_id,
                lora_r=lora_config.lora_r, lora_alpha=lora_config.lora_alpha, lora_dropout=lora_config.lora_dropout,
                lora_target_modules=",".join(lora_config.lora_target_modules),
            )
        else:
            n_layer, n_head, n_embd, block_size = req.n_layer or 8, req.n_head or 8, req.n_embd or 512, req.block_size or 512

            if req.init_from_run_id:
                source = db.get_training_run(req.init_from_run_id)
                if source is None:
                    raise HTTPException(status_code=404, detail="unknown init_from_run_id")
                if not source["latest_checkpoint_id"]:
                    raise HTTPException(status_code=400, detail="init_from_run_id run has no checkpoint yet")
                source_shape = (source["n_layer"], source["n_head"], source["n_embd"], source["block_size"])
                if source_shape != (n_layer, n_head, n_embd, block_size):
                    raise HTTPException(
                        status_code=400,
                        detail="init_from_run_id model shape does not match this run's n_layer/n_head/n_embd/block_size",
                    )
                initial_state_bytes = db.get_checkpoint_blob(source["latest_checkpoint_id"])
            else:
                torch.manual_seed(1337)  # every node must start round 0 from the identical initialization
                model = GPT(GPTConfig(vocab_size=VOCAB_SIZE, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd))
                initial_state_bytes = state_dict_to_bytes(model.state_dict())

            run_id = db.create_training_run(
                name=req.name, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size, vocab_size=VOCAB_SIZE, dataset_name=req.dataset_name,
                inner_lr=req.inner_lr, inner_steps_h=req.inner_steps_h, inner_batch_size=req.inner_batch_size,
                outer_lr=req.outer_lr, outer_momentum=req.outer_momentum, max_rounds=req.max_rounds,
                round_timeout_s=req.round_timeout_s, quorum=req.quorum, model_backend=req.model_backend,
            )

        checkpoint_id = db.save_checkpoint(run_id, "global_weights", initial_state_bytes, round_num=0)
        db.advance_run_round(run_id, 0, checkpoint_id, None)
        log_event(logger, "training_run_created", run_id=run_id, name=req.name, model_backend=req.model_backend)
        return TrainingRunCreateResponse(run_id=run_id)

    @app.post("/training/runs/{run_id}/start")
    def start_training_run(run_id: str) -> dict:
        try:
            start_run(db, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True}

    @app.get("/training/runs", response_model=list[TrainingRunOut])
    def list_training_runs() -> list[TrainingRunOut]:
        return [_training_run_to_out(r) for r in db.list_training_runs()]

    @app.get("/training/runs/{run_id}", response_model=TrainingRunOut)
    def get_training_run(run_id: str) -> TrainingRunOut:
        run = db.get_training_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return _training_run_to_out(run)

    @app.get("/training/runs/{run_id}/round/current", response_model=TrainingRoundOut)
    def get_current_training_round(
        run_id: str, node_id: str, authorization: Optional[str] = Header(None),
    ) -> TrainingRoundOut:
        _auth_node(node_id, authorization)
        run = db.get_training_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        round_row = db.get_current_round(run_id)
        if round_row is None:
            raise HTTPException(status_code=404, detail="run has not started")
        already = any(d["node_id"] == node_id for d in db.list_deltas_for_round(round_row["id"]))
        return TrainingRoundOut(
            round_id=round_row["id"], round_num=round_row["round_num"], status=round_row["status"],
            base_checkpoint_id=round_row["base_checkpoint_id"], deadline_at=round_row["deadline_at"],
            already_submitted=already, inner_steps_h=run["inner_steps_h"], inner_lr=run["inner_lr"],
            inner_batch_size=run["inner_batch_size"], block_size=run["block_size"],
        )

    @app.get("/training/checkpoints/{checkpoint_id}")
    def download_checkpoint(
        checkpoint_id: str, node_id: str, authorization: Optional[str] = Header(None),
    ) -> Response:
        _auth_node(node_id, authorization)
        blob = db.get_checkpoint_blob(checkpoint_id)
        if blob is None:
            raise HTTPException(status_code=404, detail="unknown checkpoint")
        return Response(content=blob, media_type="application/octet-stream")

    @app.post("/training/rounds/{round_id}/delta", response_model=TrainingDeltaSubmitResponse)
    async def upload_delta(
        round_id: str,
        request: Request,
        node_id: str,
        local_loss: Optional[float] = None,
        local_steps_completed: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        authorization: Optional[str] = Header(None),
    ) -> TrainingDeltaSubmitResponse:
        _auth_node(node_id, authorization)
        blob = await request.body()
        try:
            result = submit_delta(
                db, round_id, node_id, blob, local_loss, local_steps_completed, wall_time_s,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return TrainingDeltaSubmitResponse(ok=True, inserted=result["inserted"])

    @app.post("/training/runs/{run_id}/generate", response_model=GenerateResponse)
    def generate_from_run(run_id: str, req: GenerateRequest) -> GenerateResponse:
        run = db.get_training_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        if not run["latest_checkpoint_id"]:
            raise HTTPException(status_code=400, detail="run has no checkpoint yet")

        if run["model_backend"].startswith(LORA_BACKEND_PREFIX):
            base = load_frozen_base(run["base_model_id"], device="cpu", dtype=torch.float16)
            tokenizer = load_tokenizer(run["base_model_id"])
            lora_config = LoraRunConfig(
                base_model_id=run["base_model_id"], lora_r=run["lora_r"], lora_alpha=run["lora_alpha"],
                lora_dropout=run["lora_dropout"], lora_target_modules=tuple(run["lora_target_modules"].split(",")),
            )
            with temporary_lora_adapter(base, lora_config) as peft_model:
                load_adapter_state_dict(peft_model, bytes_to_state_dict(db.get_checkpoint_blob(run["latest_checkpoint_id"])))
                text = generate_chat_reply(
                    peft_model, tokenizer, req.prompt, "cpu",
                    max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                )
            return GenerateResponse(text=text)

        gpt_config = GPTConfig(
            vocab_size=run["vocab_size"], block_size=run["block_size"],
            n_layer=run["n_layer"], n_head=run["n_head"], n_embd=run["n_embd"],
        )
        model = GPT(gpt_config)
        model.load_state_dict(bytes_to_state_dict(db.get_checkpoint_blob(run["latest_checkpoint_id"])))

        encoder = get_encoder()
        is_chat = run["dataset_name"] in CHAT_DATASETS
        prompt_text = CHAT_TEMPLATE.format(instruction=req.prompt) if is_chat else req.prompt
        encoded = encoder.encode(prompt_text, allowed_special={"<|endoftext|>"})

        idx = torch.tensor([encoded], dtype=torch.long)
        stop_id = eot_token_id() if is_chat else None
        out = model.generate(
            idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature, stop_token_id=stop_id,
        )

        if is_chat:
            # Chat runs: return only the generated response, trimmed of the
            # instruction template and the trailing stop token.
            new_tokens = out[0, len(encoded):].tolist()
            if new_tokens and new_tokens[-1] == stop_id:
                new_tokens = new_tokens[:-1]
            return GenerateResponse(text=encoder.decode(new_tokens))

        # Base/pretrain runs: raw completion, prompt included, same as before.
        return GenerateResponse(text=encoder.decode(out[0].tolist()))
