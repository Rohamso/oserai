[![OserAI logo](docs/assets/oserai-mark.svg)](https://oserai.network/)

# Distributed AI Inference Network — Phase 1 PoC

[oserai.network](https://oserai.network/)

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
desktop_app/   PySide6 GUI wrapping train_agent.py -- see "Desktop app" below
scripts/       generate_test_batch.py, submit_batch.py, chaos.py,
               prepare_training_data.py, train_local.py, train_local_lora.py,
               import_local_checkpoint.py
training/      model/dataset/tokenizer/outer-optimizer code shared by the
               coordinator and every training agent -- lora.py and
               chat_dataset.py are the LoRA-on-a-real-model path, model.py
               and dataset.py are the from-scratch path
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

### Reaching the coordinator from your other devices

`--host 0.0.0.0` makes the coordinator listen on every network interface,
but that alone doesn't get a second or third machine talking to it —
they need an address that actually reaches the host, and the host's
firewall needs to let the connection through. Two ways to do that:

**Option A — Tailscale (recommended).** Install [Tailscale](https://tailscale.com)
on the coordinator host and on every contributor device, and sign them all
into the same tailnet. Each machine gets a stable `100.x.y.z` address
(`tailscale ip` on the host) that works from anywhere those devices can
reach the internet — not just the same physical LAN — with no port
forwarding or public exposure. This is also what the desktop app's build
workflow already assumes: its `coordinator_url` input example is
`http://100.x.y.z:8000`. Point every device's `coordinator_url` (desktop
app field, or `train_node_config.yaml` / `node_config.yaml`) at
`http://<host's-tailscale-ip>:8000`.

**Option B — plain LAN IP.** Find the coordinator host's LAN IP
(`ipconfig` on Windows, look for the active adapter's `IPv4 Address`; `ifconfig`
or `ip addr` on macOS/Linux) and point other devices at
`http://<host-LAN-IP>:8000` instead of `localhost`. Only works if every
device is on the same physical network.

Either way, the host's firewall still needs to allow the inbound
connection. On Windows, starting the coordinator (`uvicorn`/`python.exe`
listening on a new port) usually triggers a "Windows Defender Firewall has
blocked some features" prompt the first time — click **Allow access**, and
make sure **Private networks** is checked. Note that Windows categorizes
some home Wi-Fi networks as **Public** rather than Private (check via
`Get-NetConnectionProfile` in PowerShell) — avoid allowing the Public
profile for this, since that exposes the port to the whole network the way
a coffee shop or hotel Wi-Fi would be, not just your own devices; Tailscale
sidesteps this entirely since its interface is always Private and never
carries this exposure regardless of which Wi-Fi profile you're on.

If the prompt doesn't appear (or you already dismissed it), add the rule
manually from an **Administrator** PowerShell (`New-NetFirewallRule`
fails with "Access is denied" otherwise) — scoped to Private only:

```powershell
New-NetFirewallRule -DisplayName "OserAI coordinator" -Direction Inbound `
    -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

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
rejection, offline-node handling, crash recovery) for both the `scratch_gpt`
and `lora_*` backends — dispatch/aggregation is backend-agnostic (just moves
and averages `dict[str, Tensor]`s), so these tests use small hand-built
tensors rather than a real base model. Ollama is mocked in
`tests/test_agent_ollama.py`; torch-dependent tests are skipped automatically
if torch isn't installed — no local Ollama server, GPU, or ~3GB model
download needed to run the suite.

`tests/test_lora_smoke.py` is the one exception: a real end-to-end LoRA
round against the actual base model (load the frozen base, build the
adapter, run real local steps, round-trip the delta through the same wire
format a training agent uploads) — the only thing that actually catches a
real-world failure in that path (bad `from_pretrained` kwargs, an OOM that
crashes the process instead of raising, ...) before a contributor's machine
does. It needs a real ~3GB download (first run only, cached under
`~/.cache/huggingface` after) and the same free RAM `load_frozen_base`
itself requires, so it's excluded from the default `pytest` run (opt in via
the `smoke` marker). It's also wired into the desktop app's build workflow
as a pre-build check, not on every push -- but GitHub's standard
windows-latest runner typically doesn't have the 8GB free `load_frozen_base`
itself requires, so that CI step only fails the build for a *real* failure,
not for hitting the same RAM guard the runner itself can't clear. Run it
locally on a machine with enough free RAM for real coverage before merging
changes to `load_frozen_base`/`training/lora.py`:

```bash
pytest -m smoke tests/test_lora_smoke.py
```

## Training (DiLoCo) — distributed training, not just inference

A second, independent capability: train a model across your own
training-role nodes using [DiLoCo](https://arxiv.org/abs/2311.08105) — each
worker runs H local steps on its own shard of the run's dataset, uploads a
pseudo-gradient (weight delta) instead of raw gradients, and the coordinator
averages deltas across whoever reported and applies them via an outer
SGD+Nesterov-momentum optimizer. Same "agents always poll, coordinator never
reaches in" invariant as Phase 1 — round lifecycle is pull-based, and
`training/checkpoint_io.py`/`training/outer_optim.py` don't care what's
inside a delta — they just move and average `dict[str, Tensor]`s.

Two backends implement this, selected per-run via `model_backend`:

- **`lora_qwen2.5-1.5b`** (default) — LoRA-fine-tune a real pretrained
  instruct model. Adapts knowledge the base model already has instead of
  building it from zero, and the delta exchanged every round is the LoRA
  adapter only (~17MB) rather than the full model — both much better
  answer quality and far less bandwidth per round than training from
  scratch. **This is the path to use if you want an actually-usable chat
  model.** See "LoRA fine-tuning" below.
- **`scratch_gpt`** — the original from-scratch small-GPT path (custom
  nanoGPT-shaped architecture, GPT-2 BPE via `tiktoken`, trained from random
  init). Kept for anyone who wants to experiment with training a model's
  knowledge entirely from your own network's compute, with the ceiling that
  implies — see its own "Honest expectation" note further down.

### LoRA fine-tuning on a real model (`lora_qwen2.5-1.5b`)

Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache 2.0, ungated on Hugging Face — no account/token/license click-through
needed). The base model is always frozen; only LoRA adapter matrices (rank
16 / alpha 32 on the attention projections by default, ~4.4M params, 0.28%
of the model) ever get gradients or move over the network. The coordinator
never needs to train — it only averages adapter deltas and, for `/generate`,
runs the frozen base + adapter forward for inference.

Training data: `dolly_chat` — [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
reformatted into instruction/response pairs, tokenized through the base
model's own chat template (`tokenizer.apply_chat_template`) rather than a
hand-rolled one, since that's the format the base model was already
instruct-tuned on.

**Honest expectation**: still a small model (1.5B params) fine-tuned with
modest compute — a genuinely competent, fast, coherent chat model for
everyday questions, clearly a tier above the from-scratch path, but not
GPT-3.5-or-better. If you want frontier-model quality today, the Phase 1
inference side already runs real large pretrained chat models via Ollama;
this pipeline exists to prove out *your own* distributed fine-tuning
infrastructure on a real base model, not to out-train a frontier lab.

#### Setup

```bash
pip install -r requirements-training.txt   # torch, tiktoken, transformers, peft, accelerate, safetensors
python scripts/prepare_training_data.py    # writes training/data/dolly15k_chat.txt (also fetches tinystories.txt, used by the scratch_gpt path)
```

The base model downloads automatically the first time it's loaded (~3GB,
cached under `~/.cache/huggingface`) — same one-time-fetch idea as the
existing `ollama pull` / tiktoken steps.

Each GPU training machine needs a CUDA-matched torch wheel instead of the
default PyPI one — follow the selector at
https://pytorch.org/get-started/locally/ **before** running
`pip install -r requirements-training.txt`. Apple Silicon (`mps`) and
CPU-only both work out of the box, just slower.

#### De-risk locally before going distributed

```bash
python scripts/train_local_lora.py --steps 2000 --out out/lora_adapter.pt
python scripts/train_local_lora.py --sample-only --checkpoint out/lora_adapter.pt --prompt "What is the capital of France?"
```

#### Running a training run

```bash
curl -X POST http://localhost:8000/training/runs \
    -d '{"name": "lora-demo"}' -H 'Content-Type: application/json'
curl -X POST http://localhost:8000/training/runs/<run_id>/start

# on each contributor machine (see "Desktop app" below for a GUI alternative)
cp agent/train_node_config.example.yaml train_node_config.yaml
# edit: node_name, run_id, shard_index (0, 1, ... one per known machine), device
python -m agent.train_agent --config train_node_config.yaml
```

Chat with it the same way as any run — dashboard `Talk to a trained run` box,
or `POST /training/runs/{id}/generate` — prompts get wrapped in the chat
template and generation stops at the model's own end-of-turn token.

### Desktop app — let others train from their own PCs

`desktop_app/` is a PySide6 GUI wrapper around `agent/train_agent.py`: fill
in a coordinator URL and a run ID, hit **Start contributing**, and that
machine registers and starts pulling DiLoCo rounds in the background —
closing the window minimizes to the system tray rather than quitting, same
"install once, keep contributing" shape as BOINC/Folding@home. It drives the
exact same tested agent code as the CLI, not a separate implementation.

```bash
pip install -r requirements-desktop.txt
python -m desktop_app.main
```

To package a standalone app contributors can just double-click (no Python
install needed on their end):

```bash
cd desktop_app && pyinstaller build.spec
```

Full details, including packaging notes, in `desktop_app/README.md`. Only
verified/built on macOS so far — Windows/Linux use the same command but
need to be built on those platforms.

### From-scratch training (`scratch_gpt`)

The original path: train a small GPT-style model (nanoGPT-shaped
decoder-only transformer, GPT-2 BPE tokenizer via `tiktoken`, ~51M params at
the defaults) entirely from random init using your network's own compute —
no pretrained base model involved at all.

Two bundled datasets support a **pretrain → instruction-tune** pipeline, the
same shape real small chat models are built with:

- `tinystories` — a trimmed slice of the
  [TinyStories](https://arxiv.org/abs/2305.07759) corpus, purpose-built so
  tiny from-scratch models learn fluent, coherent English. Use this for the
  base pretrain run.
- `dolly_chat` — reformatted into an Alpaca-style
  `### Instruction:\n...\n\n### Response:\n...` template (a different,
  simpler tokenization of the same source file the LoRA path uses). Use
  this for a second, SFT run started from the pretrain run's checkpoint
  (`init_from_run_id`, see below).
- `tiny_shakespeare` still ships bundled (no fetch needed) as a fast raw-text
  smoke test — it isn't chat-formatted.

**Honest expectation**: a from-scratch model on a demo-sized corpus with
limited compute produces short, mostly-on-topic, occasionally-broken answers
in a chat *format* — clearly below the LoRA path's quality, since it's
building knowledge from zero rather than adapting a model that already has
it. Kept as an option for experimenting with pure from-scratch distributed
training; the LoRA path above is the one to use for an actually-usable
result.

#### Setup

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

#### De-risk locally before going distributed

`scripts/train_local.py` runs the same model/dataset/training loop
single-machine, no coordinator or network involved — the fast way to check
your pretrain → SFT → chat path actually works before spinning up real
training nodes:

```bash
python scripts/train_local.py --dataset tinystories --steps 3000 --out out/pretrain.pt
python scripts/train_local.py --dataset dolly_chat --init-checkpoint out/pretrain.pt --steps 3000 --out out/sft.pt
python scripts/train_local.py --sample-only --checkpoint out/sft.pt --prompt "What is the capital of France?"
```

#### Running a training run

```bash
# 1. pretrain run on TinyStories
curl -X POST http://localhost:8000/training/runs \
    -d '{"name": "pretrain", "model_backend": "scratch_gpt", "dataset_name": "tinystories"}' \
    -H 'Content-Type: application/json'
curl -X POST http://localhost:8000/training/runs/<pretrain_run_id>/start

# 2. on each GPU machine, once the pretrain run has completed (or you're
#    happy with its current checkpoint)
cp agent/train_node_config.example.yaml train_node_config.yaml
# edit: node_name, run_id, shard_index (0, 1, ... one per known machine), device
python -m agent.train_agent --config train_node_config.yaml

# 3. SFT run, seeded from the pretrain run's weights via init_from_run_id --
#    n_layer/n_head/n_embd/block_size must match the pretrain run exactly
curl -X POST http://localhost:8000/training/runs \
    -d '{"name": "sft", "model_backend": "scratch_gpt", "dataset_name": "dolly_chat", "init_from_run_id": "<pretrain_run_id>"}' \
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

#### Going bigger

Model size is a `POST /training/runs` body param (`n_layer`/`n_head`/`n_embd`/
`block_size`), not a code change. The ~51M-param default trains fine on one
consumer GPU in a demo-able window; a GPT-2-small-shaped run
(`n_layer=12, n_head=12, n_embd=768, block_size=1024`, ~124M params) needs
real GPU(s) and a lot more wall-clock/DiLoCo rounds for a noticeably better
chat experience, but needs no code changes to try. Same for swapping in a
larger slice of TinyStories (`--tinystories-mb` on the prep script).

#### Explicitly out of scope (scratch_gpt specifically)

GGUF conversion / serving the trained checkpoint through Ollama — a
from-scratch nanoGPT checkpoint isn't Ollama-loadable without real
architecture-translation and quantization work, so generation runs
server-side on the coordinator instead, and phones stay Phase 1
inference-only, untouched by this feature. (The LoRA path's base model
*is* a standard HF checkpoint, so GGUF conversion is more plausible there,
but still isn't implemented here.)

### Fault tolerance (both backends)

Quorum defaults to 1: a round closes and applies as soon as either every
currently-online training node has submitted (fast path) or
`round_timeout_s` elapses (slow path) — one dead/slow contributor never
blocks a round, and killing both workers mid-round just carries the weights
forward unchanged (`skipped`) rather than erroring, exactly like Phase 1's
"record, don't punish" verification story. `scripts/chaos.py --list --role
training` shows training-node status; since training nodes run on real
hardware (not the docker-compose demo), simulate churn by killing
`agent/train_agent.py` (or quitting the desktop app) yourself and watching
the dashboard, the same way Phase 1 says to for a real inference
contributor.

### Explicitly out of scope (training, both backends)

Multi-turn chat history (every `/generate` call is one independent
instruction/response turn, no conversation memory), on-device mobile
training, multi-concurrent-run node membership tracking (all online
training-role nodes are treated as one shared pool), and any
Byzantine/anti-cheat check on submitted deltas (two workers on different
data shards are *expected* to diverge, unlike inference's exact-match
verification) — the last point matters more once training moves from a
small set of known/trusted machines to a public downloadable app: a
malicious contributor can currently upload a garbage delta and it'll be
averaged in along with everyone else's, unnoticed.

## Explicitly out of scope (Phase 1)

Blockchain/smart contracts, token logic, payments, auth beyond the per-node
bearer token, streaming inference, model sharding, real carbon/power
measurement (only the `measured_watts`/`grid_zone` null placeholders exist),
TLS/production hardening, multi-model routing.
