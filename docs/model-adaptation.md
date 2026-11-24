# OserAI model adaptation

Colibri serves existing checkpoints. Each chosen checkpoint supplies its native
tokenizer. The local pilot uses the dense [Qwen3-0.6B Base checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)
because its architecture already runs in Colibri. The production MoE checkpoint
is a separate selection. Its tokenizer must encode the three specified control
markers directly for this ChatML recipe. [Colibri GGUF support](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/gguf.md#supported-architectures)

## Pilot inputs and reproducibility

The prose pilot uses the 61 original chapters of *Pride and Prejudice* from a
hash-checked Standard Ebooks archive. The original text was published in 1813
and the publisher dedicates its editorial contributions under CC0. The SFT
pilot uses English OpenAssistant Conversations 2 export trees with reviewed
human turns and rank-zero assistant replies. One path per tree is used. Exact
duplicates are removed before separating training and evaluation dialogues.
[Standard Ebooks edition and license](https://github.com/standardebooks/jane-austen_pride-and-prejudice/blob/a77a2dbbc0d11a17d6c68c581f17e26a65bd8168/LICENSE.md) · [OASST2 data card](https://huggingface.co/datasets/OpenAssistant/oasst2)

Install `requirements-adaptation-data.txt` and `requirements-training.txt` in
the training environment. Download `tokenizer.json`, `tokenizer_config.json`,
`config.json`, and `model.safetensors` from the pinned Qwen3 Base revision to
one directory. Use the pinned [OASST2 training Parquet](https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/data/train-00000-of-00001-88ba0162028a73fc.parquet)
and the [Standard Ebooks commit archive](https://codeload.github.com/standardebooks/jane-austen_pride-and-prejudice/tar.gz/a77a2dbbc0d11a17d6c68c581f17e26a65bd8168).

```bash
python -m scripts.prepare_adaptation_data pilot \
  --tokenizer /path/to/qwen-base/tokenizer.json \
  --book /path/to/austen.tar.gz --oasst2 /path/to/oasst2.parquet \
  --output out/adaptation/data

python -m scripts.train_adaptation_pilot \
  --base /path/to/qwen-base --data out/adaptation/data \
  --output out/adaptation/run --device mps --sft-window 160
```

Use `--resume-cpt out/adaptation/run/cpt-adapter` with a fresh output directory
to start SFT from a saved CPT adapter. Both phases use LoRA, single-example
updates and shorter local windows. Each phase reports held-out loss. These
examples are held out from adaptation and may already be known to the base
model. Broader capability and forgetting require separate evaluations.

After the pilot, merge the adapter and export one Q8_0 GGUF using a llama.cpp
checkout with its conversion dependencies installed.

```bash
python -m scripts.export_adaptation_gguf \
  --base /path/to/qwen-base --adapter out/adaptation/run/sft-adapter \
  --llama-cpp /path/to/llama.cpp --output out/adaptation/pilot-Q8_0.gguf
```

The exporter uses the existing ChatML markers and changes the serving EOS
metadata to the existing `<|im_end|>` token ID. A small pilot needs approximately
2 GB of free disk space for the temporary merged weights and final GGUF.

## Packing and provenance

For additional sources, `python -m scripts.prepare_adaptation_data pack --help`
accepts normalized JSONL. Each record has `source`, `id`, `license`, and
`origin`. CPT takes `text`, and SFT takes alternating `user` and `assistant`
messages. The reviewed catalog has `tokenizer` metadata with `model`,
`revision` and `sha256`, plus a `sources` object keyed by source. Each source
records `phase`, `slice`, `license`, `origin`, `source_url`, `revision`,
`license_evidence`, and `artifact_sha256`. A source entry must reflect the
item-level permissions and origin for its records. The packer checks claims
against the catalog, whose review establishes the provenance.

CPT filters records shorter than 50 words, uses an education score of at least
3 for `everyday_web`, checks code licenses and languages for `logic_code`, and
requires CC0 or CC BY 4.0 for `stem_health`. It applies exact and approximate
MinHash LSH duplicate filtering. Recognized emails, phone numbers and US
social security numbers are redacted. At least three matches at three or more
per 100 words cause rejection. Broader PII detection and review are needed
before redistribution.

CPT packs 8,192-token rows separated by the native `<|endoftext|>` token.
Incomplete CPT tail tokens are counted in the manifest. SFT retains whole
ChatML conversations and masks system and user tokens with `-100`. The
assistant text and end-of-turn marker contribute to loss. Each uncompressed
Parquet shard lists referenced source records and offsets, and its manifest
records its SHA-256, token and row counts, and source metadata. The shard
target is about 1.5 GB, with a smaller final shard for each split.

## Source decisions for the production corpus

| Proposed source | Decision under the directive's allowlist |
| --- | --- |
| FineWeb-Edu and Dolma v1.7 | Exclude their bulk releases. Their collection license is ODC-BY. Find independently reviewed sources for this token allocation. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#license) · [Dolma](https://huggingface.co/datasets/allenai/dolma) |
| Wikimedia dumps and Dolly-15k | Exclude the default share-alike releases. A source or item with separate allowlisted rights would need separate provenance. [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| The Stack v2 | Consider only file revisions with verified original MIT, Apache 2.0, or BSD-2/3-Clause licenses. Preserve repository, revision, path, and attribution. [The Stack v2 data card](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| Standard Ebooks and Project Gutenberg | Review each specific edition and jurisdiction. The pilot approves the original Austen text from one pinned archive. |
| PMC and arXiv | Review the item-level license and use the CC0 or CC BY 4.0 subset. [PMC licenses](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) · [arXiv licenses](https://info.arxiv.org/help/license/index.html) |
| Tulu 3 SFT Mix | Exclude the bulk release. It uses ODC-BY and includes noncommercial and third-party-generated subsets. Review any independently sourced upstream subset before inclusion. [Tulu 3 data card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| OpenAssistant Conversations 2 | The published ready-for-export archive has 13,854 trees and 135,174 messages. Count actual deduplicated dialogues after selection. [OASST2 data card](https://huggingface.co/datasets/OpenAssistant/oasst2#ready-for-export-trees) |
| HelpSteer2 | The publisher lists 21,362 response samples, with paired responses per prompt and publisher-generated output. Review the data card and select suitable answers per prompt before assigning dialogue targets. [HelpSteer2 data card](https://huggingface.co/datasets/nvidia/HelpSteer2) |

The proposed SFT line items sum to 700,000 even before source exclusions and
deduplication. Set new allocations after measuring eligible records against
the 750,000-dialogue goal. The license checks support an audit trail, and
redistribution decisions still depend on the original source and item terms.

## Production training and release

`training/adaptation_recipe.json` records the target hyperparameters. The
50-billion-token plan requires about 200 GB for 32-bit token IDs alone. Run
production CPT and SFT on a training cluster. Qwen3-30B-A3B-Base is an
Apache-2.0 MoE candidate because the Qwen3 MoE architecture is supported by
Colibri. Review the checkpoint's upstream training provenance, native tokenizer,
and full-precision training support before selecting it for production.
[MoE model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) · [Colibri GGUF support](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/gguf.md#supported-architectures)

Approve replay samples from training data matching the original distribution
and hold validation data out of training. Decide whether the 5% replay is
included in the 50-billion-token total or added to the five slice targets.

For a production release, export unquantized safetensors, AWQ and GPTQ for
compatible accelerator runtimes, and Q4_K_M and Q8_0 GGUF for CPU inference.
Verify loading and generation for each export. Before publishing shards to
IPFS or Arweave, verify redistribution rights and PII screening. Record the
resulting root CID, model commit, and training license manifest through the
network's governance mechanism when it is available.
