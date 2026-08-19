"""Pydantic models for every coordinator API payload."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class NodeRegisterRequest(BaseModel):
    name: str
    model_name: str
    max_concurrent_jobs: int = 1
    city: Optional[str] = None
    country: Optional[str] = None
    hardware: Optional[str] = None
    measured_watts: Optional[float] = None
    grid_zone: Optional[str] = None


class NodeRegisterResponse(BaseModel):
    node_id: str
    token: str


class HeartbeatRequest(BaseModel):
    status: str = Field(description="idle or busy")
    cpu_pct: float = 0.0
    ram_free_mb: float = 0.0


class NodeOut(BaseModel):
    id: str
    name: str
    city: Optional[str]
    country: Optional[str]
    hardware: Optional[str]
    model_name: str
    role: str
    max_concurrent_jobs: int
    measured_watts: Optional[float]
    grid_zone: Optional[str]
    status: str
    last_heartbeat: Optional[float]
    cpu_pct: Optional[float]
    ram_free_mb: Optional[float]
    jobs_done: int
    jobs_failed: int
    tokens_total: int
    duration_total: float
    mismatch_count: int
    requeue_count: int
    tokens_per_sec_avg: float


class BatchCreateRequest(BaseModel):
    prompts: list[str]
    model_name: str
    max_tokens: int = 256
    timeout_s: int = 300
    verification_pct: Optional[float] = None  # None -> coordinator default


class BatchCreateResponse(BaseModel):
    batch_id: str
    total_jobs: int


class JobNextResponse(BaseModel):
    job_id: str
    prompt: str
    max_tokens: int
    timeout_s: int
    model_name: str


class JobResultRequest(BaseModel):
    node_id: str
    success: bool
    output_text: str = ""
    output_sha256: str = ""
    tokens_generated: int = 0
    duration_s: float = 0.0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None


class BatchProgress(BaseModel):
    batch_id: str
    model_name: str
    total_jobs: int
    queued: int
    dispatched: int
    running: int
    completed: int
    verified: int
    mismatched: int
    failed: int
    retry_count: int
    throughput_tokens_per_sec: float
    eta_seconds: Optional[float]


class NodeStats(BaseModel):
    node_id: str
    name: str
    role: str
    status: str
    jobs_done: int
    jobs_failed: int
    tokens_per_sec_avg: float
    failure_rate: float
    mismatch_count: int
    uptime_pct: float


class StatsResponse(BaseModel):
    nodes: list[NodeStats]
    batches: list[BatchProgress]
    aggregate_tokens_per_sec: float
    cumulative_payout: float


# ---------------------------------------------------------------------- #
# training (DiLoCo)
# ---------------------------------------------------------------------- #

class TrainingNodeRegisterRequest(BaseModel):
    name: str
    hardware: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class TrainingRunCreateRequest(BaseModel):
    name: str
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    dataset_name: str = "tinystories"
    # Set to a completed run's id to seed round 0 from that run's final
    # checkpoint instead of a fresh random init -- e.g. start an SFT run
    # (dataset_name="dolly_chat") from a finished pretrain run's weights.
    # n_layer/n_head/n_embd/block_size must match the source run exactly.
    init_from_run_id: Optional[str] = None
    inner_lr: float = 1e-3
    inner_steps_h: int = 50
    inner_batch_size: int = 64
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    max_rounds: int = 200
    round_timeout_s: int = 600
    quorum: int = 1


class TrainingRunCreateResponse(BaseModel):
    run_id: str


class TrainingRunOut(BaseModel):
    id: str
    name: str
    status: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dataset_name: str
    inner_lr: float
    inner_steps_h: int
    inner_batch_size: int
    outer_lr: float
    outer_momentum: float
    max_rounds: int
    round_timeout_s: int
    quorum: int
    current_round: int
    created_at: float
    completed_at: Optional[float]
    latest_checkpoint_id: Optional[str]


class TrainingRoundOut(BaseModel):
    round_id: str
    round_num: int
    status: str
    base_checkpoint_id: str
    deadline_at: float
    already_submitted: bool
    inner_steps_h: int
    inner_lr: float
    inner_batch_size: int
    block_size: int


class TrainingDeltaSubmitResponse(BaseModel):
    ok: bool
    inserted: bool


class GenerateRequest(BaseModel):
    prompt: str = "\n"
    max_new_tokens: int = 300
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    text: str
