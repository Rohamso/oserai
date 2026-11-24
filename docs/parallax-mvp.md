# OserAI with Gradient Parallax

This MVP connects OserAI's batch workflow to a Parallax inference cluster.
Use one model and two or three trusted machines for the first cluster test.

For the public chat website, see the [cPanel guide](chat-cpanel.md). The Razer
Windows laptop's RTX 2070 currently uses the separate [Ollama host setup](chat-windows.md);
that path does not establish Parallax compatibility on this older GPU.

```text
Batch client -> OserAI coordinator -> OserAI bridge agent
                                       |
                                Parallax scheduler
                                  /           \
                            worker A       worker B
```

Run **one bridge per cluster**, not one bridge on every worker. The bridge
registers as a single inference node in OserAI. Parallax owns worker membership,
shard placement and inference scheduling; OserAI owns customer jobs and retries.
The current bridge processes one request at a time. Its usage and simulated
ledger entries belong to the cluster, with no per-worker contribution split.

## 1. Start Parallax separately

Follow Gradient's [installation guide](https://github.com/GradientHQ/parallax/blob/main/docs/user_guide/install.md).
Parallax's current source requires Python >=3.11 and <3.14. Use a separate
checkout/environment from OserAI so its GPU/MLX dependencies do not replace
OserAI's dependencies. macOS source installs target Apple Silicon; see Gradient's
instructions for Linux GPUs and the Windows application.

In that checkout, the documented two-worker example is:

```bash
parallax run -m Qwen/Qwen3-0.6B -n 2
```

On **each** worker, including the scheduler host if it also contributes compute,
activate the Parallax environment in a separate terminal. On the same local
network, run:

```bash
source .venv/bin/activate
parallax join
```

For workers on different networks, use the scheduler address printed by Parallax:

```bash
parallax join -s <scheduler-address-from-Parallax>
```

Wait for the workers and model to become ready. Keep the scheduler and workers
running during the following steps. Use Parallax's UI for physical worker status.
Parallax documents workers for macOS, Linux, and Windows. Phones can use the
OserAI Chat website in their browsers. The Parallax pilot needs computers to
contribute compute.

For an initial **single-machine** check, use `parallax serve -m Qwen/Qwen3-0.6B`
instead. Its default HTTP port is **3000**; the scheduler's default is **3001**.
Set the bridge's `parallax_url` to the corresponding address.

Check the scheduler API from the bridge machine:

```bash
curl --fail http://localhost:3001/v1/models
```

The returned `data[].id` must match the bridge and batch `model_name` exactly.
The bridge checks this before each completion, because the scheduler can
substitute its configured model for the requested model. A model listing is not
a guarantee that all workers are ready; a successful completion is the readiness
check. Use one fixed model throughout a test.

## Troubleshoot worker discovery

1. Check that the scheduler is still running and the worker is connected to
   the same local network. On that network, run `parallax join` from the activated
   Parallax environment. For a worker on another network, use the address shown
   by the scheduler with `parallax join -s <scheduler-address>`.
2. On macOS, open **System Settings → Privacy & Security → Local Network**
   and allow the app running your terminal. This is relevant if discovery fails
   or Parallax reports `No route to host`.
3. Check whether the local firewall permits Parallax traffic between the trusted
   machines. Parallax uses TCP and UDP for its worker connections. See
   [Gradient's networking FAQ](https://github.com/GradientHQ/parallax/blob/main/docs/user_guide/quick_start.md#faq)
   for port details.
4. When the workers connect, wait for the model to load and try a real chat
   completion. A model entry from `/v1/models` alone does not verify that workers
   can generate replies.

## 2. Start an isolated OserAI coordinator

From the OserAI repository, create a clean environment if needed:

```bash
python3.12 -m venv .venv-parallax-mvp
source .venv-parallax-mvp/bin/activate
pip install -r requirements.txt
mkdir -p out
COORDINATOR_DB_PATH=out/parallax-mvp.db uvicorn coordinator.main:app --host 127.0.0.1 --port 8000
```

The separate database leaves prior experiment data available. When opening an
older database, the coordinator adds backend columns automatically and assigns
existing nodes and batches to Ollama.

## 3. Start the bridge

In another terminal at the OserAI repository, activate the OserAI environment:

```bash
source .venv-parallax-mvp/bin/activate
cp agent/parallax_node_config.example.yaml parallax_node_config.yaml
python -m agent.main --config parallax_node_config.yaml
```

Edit the config if the scheduler runs elsewhere. `parallax_url` accepts a base
URL with or without `/v1`. `model_name` is the model exposed by `/v1/models`,
not an Ollama model tag. Environment overrides include `AGENT_PARALLAX_URL`,
`AGENT_INFERENCE_BACKEND` and `AGENT_MODEL_NAME`.

The example uses its own ignored state file. Restarting with the same config
reuses its registration. Changing the backend, model or endpoint retires the
old registration and registers again. Use a separate state file for each agent.

## 4. Submit a batch and read answers

In another activated OserAI terminal:

```bash
python scripts/generate_test_batch.py --count 10 --output out/mvp-prompts.jsonl
python scripts/submit_batch.py out/mvp-prompts.jsonl \
  --inference-backend parallax \
  --model-name Qwen/Qwen3-0.6B \
  --max-tokens 128
```

The command prints the results URL. Open it or request it with:

```bash
curl --fail http://localhost:8000/batches/<batch_id>/jobs
```

Each primary job includes its status, prompt, answer, generated-token count,
duration and error, if any. The dashboard at **http://localhost:8000/** shows
the backend, cluster usage, batch progress and a link to results.

API clients select the same route explicitly:

```json
{
  "model_name": "Qwen/Qwen3-0.6B",
  "inference_backend": "parallax",
  "prompts": ["Explain the water cycle in three sentences."],
  "max_tokens": 128
}
```

Omitting `inference_backend` keeps the existing Ollama behavior. Dispatch matches
both backend and model, even when two backends use the same model name.

Parallax batches default to **zero verification duplicates**, regardless of the
coordinator's Ollama verification default. Explicit nonzero `verification_pct`
is rejected. Outputs from heterogeneous runtimes are not guaranteed to match
byte for byte, and two bridge processes may reach the same underlying workers.
Completed Parallax jobs remain `completed`; they are not marked `verified`.

The bridge uses non-streaming `/v1/chat/completions`, requests deterministic
sampling, and disables optional thinking for this short-answer baseline. It
requires a text answer and `usage.completion_tokens`. HTTP errors, timeouts,
wrong models and malformed completions enter OserAI's existing retry/failure
workflow and do not create successful usage entries. `duration_s` measures the
whole request, including the model check, rather than GPU decode time alone.

## 5. Benchmark before expanding

The direct benchmark calls the configured backend without creating batch jobs
or ledger entries:

```bash
python -m scripts.benchmark_inference out/mvp-prompts.jsonl \
  --config parallax_node_config.yaml \
  --max-tokens 128 --output out/parallax-benchmark.json

python -m scripts.benchmark_inference out/mvp-prompts.jsonl \
  --config node_config.yaml \
  --max-tokens 128 --output out/ollama-benchmark.json
```

The Ollama command needs a normal Ollama node config and a running Ollama
server. Use equivalent weights/quantization where possible, identical prompts
and token limits, and record the hardware/network and exact model variants.
Different model sizes are a product comparison, not an engine speed comparison.
Warm up each backend first and repeat the benchmark; cold model loading can
dominate the first request. Reports include successes, failures, median request
latency, tokens, overall throughput and individual answers for quality review.
Failure exits with status 1 and still writes the report.

For the first real-cluster acceptance test:

1. Complete ten prompts on two or three workers and inspect the answers.
2. Confirm the dashboard and ledger show one cluster's usage.
3. Stop a worker during a batch; record whether Parallax recovers or OserAI
   retries/fails the request. Restore the worker and submit another batch.
4. Save benchmark reports and compare latency, answer quality and reliability
   with the Ollama baseline before expanding desktop onboarding.

## Validation and limits

Run `python -m pytest` for adapter, API lifecycle, backend routing, database
migration and retry tests. These mock the Parallax HTTP service and need no
model download. They do **not** establish real GPU/MLX compatibility, performance
or multi-machine recovery; those require the real-cluster acceptance test above.

The API contract was checked against Gradient Parallax source revision
`162354a03234a28cf6e2946e2e0b2203da7c3721`:
[scheduler](https://github.com/GradientHQ/parallax/blob/162354a03234a28cf6e2946e2e0b2203da7c3721/src/backend/main.py),
[request forwarding](https://github.com/GradientHQ/parallax/blob/162354a03234a28cf6e2946e2e0b2203da7c3721/src/backend/server/request_handler.py).
This is a reference for the adapter contract, not a claim that this revision
has been benchmarked on these machines.

This remains a trusted-machine MVP using the repository's existing access
model. Per-worker payments, public contributor verification and automatic
Parallax installation in the desktop trainer are separate work. Training still
uses OserAI's existing DiLoCo/LoRA path.
