# Contributing to OserAI

OserAI welcomes reproductions, documentation fixes, and small patches that
make it easier to run a node or understand its results.

## Pick a first task

- **Test an inference worker.** Run the [one-job quickstart](README.md#run-a-first-inference-job).
  Share the OS, CPU or GPU, RAM, runtime and model version, prompt, and outcome.
  For a timed run, use the [benchmark commands](docs/parallax-mvp.md#5-benchmark-before-expanding).
- **Test chat.** Send a few prompts through a configured chat gateway. Report
  the question, the reply you observed, what you expected, and the browser
  you used. Use sample prompts you can share in an issue.
- **Help with training.** Compare training time or held-out loss on the
  [local adaptation pilot](docs/model-adaptation.md), or help connect its
  reviewed ChatML shards to the [desktop training agent](desktop_app/README.md).
  Live training rounds currently use invited, trusted devices.

## Develop locally

Create a virtual environment from the repository root and run the core tests.
The default suite uses mocks and small fixtures for inference. It excludes the
model-download smoke test.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-chat.txt
python -m pytest -q
```

Install `requirements-training.txt` for checkpoint and training tests, and
`requirements-adaptation-data.txt` for the data packer tests. PostgreSQL-backed
chat tests use `OSERAI_TEST_POSTGRES_URL` when it points at a disposable test
database.

Keep pull requests focused and describe how you tested the behavior. Include
the current output and a short reproduction for bug fixes. Remove access
codes, API keys, private prompts, and machine-specific configuration from logs
or examples you share.

For changes involving model or dataset inputs, include the source, version,
license, and provenance evidence. Keep generated checkpoints and datasets
out of source commits.
