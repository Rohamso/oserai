"""Pydantic models for every coordinator API payload."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from common.inference import InferenceBackend


class NodeRegisterRequest(BaseModel):
    name: str
    model_name: str
    inference_backend: InferenceBackend = "ollama"
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
    inference_backend: InferenceBackend = "ollama"
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
    inference_backend: InferenceBackend = "ollama"
    max_tokens: int = 256
    timeout_s: int = 300
    verification_pct: Optional[float] = Field(default=None, ge=0, le=1)


class BatchCreateResponse(BaseModel):
    batch_id: str
    total_jobs: int


class JobNextResponse(BaseModel):
    job_id: str
    prompt: str
    max_tokens: int
    timeout_s: int
    model_name: str
    inference_backend: InferenceBackend = "ollama"


class JobOut(BaseModel):
    id: str
    status: str
    prompt: str
    output_text: Optional[str]
    tokens_generated: Optional[int]
    duration_s: Optional[float]
    tokens_per_sec: Optional[float]
    error: Optional[str]


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
    inference_backend: InferenceBackend = "ollama"
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
    inference_backend: InferenceBackend = "ollama"
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
    # "lora_qwen2.5-1.5b" (default): LoRA-fine-tune a real pretrained
    # instruct model -- see training/lora.py. "scratch_gpt": the original
    # from-scratch small-GPT path (training/model.py), kept for anyone who
    # wants to experiment with it, no longer the default now that LoRA
    # gives real answer quality at far less bandwidth per round.
    model_backend: str = "lora_qwen2.5-1.5b"

    # n_layer/n_head/n_embd: scratch_gpt only (from-scratch model
    # architecture). block_size (max local-step sequence length) applies to
    # both backends. All None pick up the backend-specific defaults applied
    # in the /training/runs handler.
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    block_size: Optional[int] = None

    # lora_* backends only -- which pretrained model to fine-tune and the
    # adapter's shape. None picks up training/lora.py's defaults.
    base_model_id: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[list[str]] = None

    dataset_name: str = "dolly_chat"
    # Set to a completed run's id to seed round 0 from that run's final
    # checkpoint instead of a fresh init -- e.g. continue a scratch_gpt SFT
    # run from a finished pretrain run's weights. scratch_gpt only: an
    # architecture match is required (n_layer/n_head/n_embd/block_size).
    # Not used for lora_* backends -- every LoRA run already starts from
    # the same pretrained base_model_id, there's no separate pretrain stage.
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
    model_backend: str
    n_layer: Optional[int]
    n_head: Optional[int]
    n_embd: Optional[int]
    block_size: Optional[int]
    vocab_size: int
    base_model_id: Optional[str]
    lora_r: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]
    lora_target_modules: Optional[str]
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
