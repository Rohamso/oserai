# Distributed AI Inference Network — Phase 1 PoC

Prove that mixed consumer hardware (gaming PCs, Macs, laptops, Raspberry Pi)
can serve batch LLM inference jobs reliably, with result verification and
full performance logging. No blockchain, no tokens, no payments — a SQLite
table simulates the ledger that a real chain would replace in a later phase.

## Architecture

```
                         ┌────────────────────────────┐
                         │        Coordinator          │
                         │        (FastAPI + SQLite)   │
                         │                              │
                         │  node registry   job queue   │
                         │  dispatch loop   verification │
                         │  ledger          stats/dash   │
                         └───────────▲──────────▲───────┘
                                     │          │
                         plain HTTP  │          │  plain HTTP
                         polling     │          │  polling
                  ┌──────────────────┘          └──────────────────┐
                  │                                                 │
        ┌─────────┴─────────┐                             ┌────────┴──────────┐
        │   Node agent A     │      ...                    │   Node agent N     │
        │  (gaming PC, Mac,  │                              │  (Raspberry Pi,    │
        │   laptop, ...)     │                              │   laptop, ...)     │
        │                    │                              │                    │
        │  register/heartbeat│                              │  register/heartbeat│
        │  poll /jobs/next   │                              │  poll /jobs/next   │
        │  run on Ollama     │                              │  run on Ollama     │
        │  (localhost:11434) │                              │  (localhost:11434) │
        │  POST /jobs/result │                              │  POST /jobs/result │
        └────────────────────┘                              └────────────────────┘
```

Agents always initiate contact (register, heartbeat, poll for work, post
results) — the coordinator never opens a connection into a contributor's
network, so there's nothing to punch through NAT for. A background loop on
the coordinator scores idle nodes against queued jobs (`score_node()`) and
pre-assigns work; agents just ask "is there anything assigned to me?" on
every poll.

## Repo layout

```
common/        shared JSON-line logging + sha256 helper
coordinator/   FastAPI app, SQLite data layer, dispatch/scoring/verification,
               training round dispatch (training_dispatch.py), dashboard
agent/         node agent: config, Ollama client, coordinator client, main
               loop; train_agent.py is the separate DiLoCo training node loop
scripts/       generate_test_batch.py, submit_batch.py, chaos.py,
               prepare_training_data.py, train_local.py
training/      model/dataset/tokenizer/outer-optimizer code shared by the
               coordinator and every training agent
tests/         pytest suite (dispatch scoring, verification, requeue,
               heartbeat expiry, DiLoCo round lifecycle)
```

## Data model (SQLite, `coordinator/db.py`)

- `nodes` — registry + lifetime stats (jobs done/failed, tokens, mismatch count, requeue count)
- `batches` — one row per submitted batch (model, max_tokens, timeout, verification %)
- `jobs` — one row per unit of work; `is_duplicate` + `verification_group_id` link a verification pair
- `job_events` — append-only state-transition log (queued → dispatched → running → completed/failed/requeued/verified/mismatched)
- `ledger` — append-only, one row per completed job: tokens, duration, `hypothetical_payout`

All SQL lives in `coordinator/db.py`'s `Database` class — nothing else in the
codebase writes raw SQL, so swapping SQLite for Postgres later means
replacing that one file's connection/query internals, not the call sites.

## Setup — coordinator

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn coordinator.main:app --host 0.0.0.0 --port 8000   # coordinator_config.yaml already ships with sane defaults
```

Dashboard: http://localhost:8000/

## Setup — a node

On the contributor machine, make sure [Ollama](https://ollama.com) is
installed and serving on `localhost:11434`, then pull the model you intend
to serve:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

Then:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp agent/node_config.example.yaml node_config.yaml
# edit 4 lines: node_name, coordinator_url, model_name, hardware
python -m agent.main --config node_config.yaml
```

The agent stores its `node_id`/token in `.node_state.json` next to the
config and reuses them on restart. `Ctrl-C` / `SIGTERM` finishes or fails
whatever job is in flight, deregisters, and exits.

## Submitting work

```bash
python scripts/generate_test_batch.py --count 50 --output test_batch.jsonl
python scripts/submit_batch.py test_batch.jsonl \
    --coordinator-url http://localhost:8000 \
    --model-name llama3.1:8b-instruct-q4_K_M \
    --verification-pct 0.2
```

`submit_batch.py` tails `/batches/{id}` and prints a live progress line
until the batch completes.

## Docker-compose demo (no real hardware needed)

Runs the coordinator plus 3 fake agents on one machine, all pointed at your
host's Ollama:

```bash
ollama pull llama3.2:1b   # or set DEMO_MODEL to whatever you have pulled
ollama serve              # if not already running

DEMO_MODEL=llama3.2:1b docker compose up --build
```

Then, from another terminal:

```bash
python scripts/generate_test_batch.py --count 50 --output test_batch.jsonl
python scripts/submit_batch.py test_batch.jsonl --model-name llama3.2:1b --verification-pct 0.2
```

Watch it live at http://localhost:8000/.

To demo churn/requeue handling, kill one of the fake agents mid-batch:

```bash
python scripts/chaos.py               # picks a random live agent and kills its container
python scripts/chaos.py --list        # just show current node status
```

or manually: `docker compose kill agent2`. Within `heartbeat_timeout_s`
(default 90s) that node flips to `offline` on the dashboard, its in-flight
job requeues to another node, and the batch still completes — check the
`retry_count` in the batch progress line.

## Reading the dashboard

- **Top summary**: aggregate network tokens/sec, nodes online/total,
  cumulative simulated payout (Σ `tokens_generated × payout_rate` across the
  ledger).
- **Active Batches**: progress bar (completed+verified+mismatched+failed vs.
  total submitted prompts), retry count, ETA, live throughput.
- **Nodes table**: status (idle/busy/offline), lifetime jobs done/failed,
  average tokens/sec, failure rate, mismatch count, uptime.

It polls `GET /stats` every 5 seconds; no build step, no framework.

## Verification sampling

A configurable percentage of jobs (`default_verification_pct`, 5% by
default; pass `verification_pct` on `POST /batches` to override — the demo
above uses 20%) are silently duplicated to a second node running the same
model. Both agents run with `temperature=0` and `seed=42`, so identical
prompts on an identical model tag should produce byte-identical output. When
both halves land, the coordinator compares `sha256(output)`:

- match → both jobs marked `verified`
- mismatch → both jobs marked `mismatched`, `mismatch_count` incremented on
  both nodes, and the full pair (both outputs + hashes) is logged for
  inspection

There's no slashing — Phase 1 only records and exposes mismatches.

## Tests

```bash
pytest
```

Covers dispatch scoring (`score_node`, `pick_best_node`, verification-aware
node exclusion), verification-pair comparison, requeue-on-timeout and
terminal failure after `max_job_attempts`, heartbeat expiry, and the DiLoCo
training-round lifecycle (fast/slow close, skipped rounds, duplicate-delta
rejection, offline-node handling, crash recovery). Ollama is mocked in
`tests/test_agent_ollama.py`; torch-dependent tests are skipped automatically
if torch isn't installed — no local Ollama server or GPU needed to run the
suite.

## Training (DiLoCo) — distributed training, not just inference

A second, independent capability: train a small GPT-style model (nanoGPT-shaped
decoder-only transformer, GPT-2 BPE tokenizer via `tiktoken`, ~51M params at
the defaults) across your own training-role nodes using
[DiLoCo](https://arxiv.org/abs/2311.08105) — each worker runs H local AdamW
steps on its own shard of the run's dataset, uploads a pseudo-gradient
(weight delta) instead of raw gradients, and the coordinator averages deltas
across whoever reported and applies them via an outer SGD+Nesterov-momentum
optimizer. Same "agents always poll, coordinator never reaches in" invariant
as Phase 1 — round lifecycle is pull-based.

Two bundled datasets support a **pretrain → instruction-tune** pipeline, the
same shape real small chat models are built with:

- `tinystories` — a trimmed slice of the
  [TinyStories](https://arxiv.org/abs/2305.07759) corpus, purpose-built so
  tiny from-scratch models learn fluent, coherent English. Use this for the
  base pretrain run.
- `dolly_chat` — [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  reformatted into an Alpaca-style `### Instruction:\n...\n\n### Response:\n...`
  template. Use this for a second, SFT run started from the pretrain run's
  checkpoint (`init_from_run_id`, see below).
- `tiny_shakespeare` still ships bundled (no fetch needed) as a fast raw-text
  smoke test — it isn't chat-formatted.

**Honest expectation**: this is still a from-scratch model trained on a
demo-sized corpus with limited compute. A fully trained pretrain+SFT run
produces short, mostly-on-topic, occasionally-broken answers in a chat
*format* — not GPT-3.5-or-better quality. If you want an actually
GPT-quality chat experience today, the Phase 1 inference side already runs
real pretrained chat models via Ollama; this pipeline exists to prove out
*your own* distributed training infrastructure, not to compete with them.

### Setup

The coordinator only ever does CPU-side aggregation and CPU-side text
generation — it never trains — so a plain CPU `pip install torch` is enough
there regardless of platform:

```bash
pip install -r requirements-training.txt
```

Each GPU training machine needs a CUDA-matched wheel instead of the default
PyPI one — follow the selector at https://pytorch.org/get-started/locally/
**before** running the same `pip install -r requirements-training.txt`. If
torch isn't installed, the coordinator logs a warning and simply doesn't
register any `/training/*` routes — a pure Phase 1 inference deployment
never needs torch at all.

`tiktoken` fetches the GPT-2 BPE encoder ranks file over the network the
first time it's used (cached under `~/.cache/tiktoken` afterward) — a
one-time step, same idea as the existing one-time `ollama pull`.

Fetch and format the two chat-pipeline corpora once, on any machine with
network access (they only need to exist wherever the coordinator and
training agents read `training/data/` from — copy the files over if that's
not the same machine that ran the script):

```bash
python scripts/prepare_training_data.py   # writes training/data/{tinystories,dolly15k_chat}.txt
```

### De-risk locally before going distributed

`scripts/train_local.py` runs the same model/dataset/training loop
single-machine, no coordinator or network involved — the fast way to check
your pretrain → SFT → chat path actually works before spinning up real
training nodes:

```bash
python scripts/train_local.py --dataset tinystories --steps 3000 --out out/pretrain.pt
python scripts/train_local.py --dataset dolly_chat --init-checkpoint out/pretrain.pt --steps 3000 --out out/sft.pt
python scripts/train_local.py --sample-only --checkpoint out/sft.pt --prompt "What is the capital of France?"
```

### Running a training run

```bash
# 1. pretrain run on TinyStories
curl -X POST http://localhost:8000/training/runs \
    -d '{"name": "pretrain", "dataset_name": "tinystories"}' -H 'Content-Type: application/json'
curl -X POST http://localhost:8000/training/runs/<pretrain_run_id>/start

# 2. on each GPU machine, once the pretrain run has completed (or you're
#    happy with its current checkpoint)
cp agent/train_node_config.example.yaml train_node_config.yaml
# edit: node_name, run_id, shard_index (0, 1, ... one per known machine), device
python -m agent.train_agent --config train_node_config.yaml

# 3. SFT run, seeded from the pretrain run's weights via init_from_run_id --
#    n_layer/n_head/n_embd/block_size must match the pretrain run exactly
curl -X POST http://localhost:8000/training/runs \
    -d '{"name": "sft", "dataset_name": "dolly_chat", "init_from_run_id": "<pretrain_run_id>"}' \
    -H 'Content-Type: application/json'
curl -X POST http://localhost:8000/training/runs/<sft_run_id>/start
# point agent/train_agent.py's train_node_config.yaml run_id at the SFT run and run it again
```

Watch round progress on the dashboard (`Training Runs` section) alongside
the existing inference nodes/batches. Once the SFT run has a checkpoint
(even before it finishes), use the `Talk to a trained run` box on the
dashboard, or `POST /training/runs/{id}/generate` directly, to chat with it
— `dolly_chat` runs get their prompt wrapped in the instruction template and
generation stops at `<|endoftext|>` instead of rambling to
`max_new_tokens`; other runs return a raw text completion as before.

### Fault tolerance

Quorum defaults to 1: a round closes and applies as soon as either every
currently-online training node has submitted (fast path) or
`round_timeout_s` elapses (slow path) — one dead/slow GPU never blocks a
round, and killing both workers mid-round just carries the weights forward
unchanged (`skipped`) rather than erroring, exactly like Phase 1's "record,
don't punish" verification story. `scripts/chaos.py --list --role training`
shows training-node status; since training nodes run on real hardware (not
the docker-compose demo), simulate churn by killing `agent/train_agent.py`
yourself and watching the dashboard, the same way Phase 1 says to for a real
inference contributor.

### Going bigger

Model size is a `POST /training/runs` body param (`n_layer`/`n_head`/`n_embd`/
`block_size`), not a code change. The ~51M-param default trains fine on one
consumer GPU in a demo-able window; a GPT-2-small-shaped run
(`n_layer=12, n_head=12, n_embd=768, block_size=1024`, ~124M params) needs
real GPU(s) and a lot more wall-clock/DiLoCo rounds for a noticeably better
chat experience, but needs no code changes to try. Same for swapping in a
larger slice of TinyStories (`--tinystories-mb` on the prep script).

### Explicitly out of scope

GGUF conversion / serving the trained checkpoint through Ollama — a
from-scratch nanoGPT checkpoint isn't Ollama-loadable without real
architecture-translation and quantization work, so generation runs
server-side on the coordinator instead, and phones stay Phase 1
inference-only, untouched by this feature. Also out of scope: multi-turn
chat history (every `/generate` call is one independent instruction/response
turn, no conversation memory), on-device mobile training,
multi-concurrent-run node membership tracking (all online training-role
nodes are treated as one shared pool), and any Byzantine/anti-cheat check on
submitted deltas (two workers on different data shards are *expected* to
diverge, unlike inference's exact-match verification).

## Explicitly out of scope (Phase 1)

Blockchain/smart contracts, token logic, payments, auth beyond the per-node
bearer token, streaming inference, model sharding, real carbon/power
measurement (only the `measured_watts`/`grid_zone` null placeholders exist),
TLS/production hardening, multi-model routing.
