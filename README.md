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
coordinator/   FastAPI app, SQLite data layer, dispatch/scoring/verification, dashboard
agent/         node agent: config, Ollama client, coordinator client, main loop
scripts/       generate_test_batch.py, submit_batch.py, chaos.py
tests/         pytest suite (dispatch scoring, verification, requeue, heartbeat expiry)
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
terminal failure after `max_job_attempts`, and heartbeat expiry. Ollama is
mocked in `tests/test_agent_ollama.py` — no local Ollama server needed to
run the suite.

## Explicitly out of scope (Phase 1)

Blockchain/smart contracts, token logic, payments, auth beyond the per-node
bearer token, streaming inference, model sharding, real carbon/power
measurement (only the `measured_watts`/`grid_zone` null placeholders exist),
TLS/production hardening, multi-model routing.
