"""Pack reviewed JSONL inputs or prepare the reproducible OserAI adaptation pilot."""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from training.adaptation_data import build_corpus, clean_messages, sha256


OASST_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
OASST_SHA256 = "9638227b41b0a4fb69afa5d5c4b52e3fb6abfd56c9870d81bae32c9981e769b1"
QWEN_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
QWEN_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
AUSTEN_REVISION = "a77a2dbbc0d11a17d6c68c581f17e26a65bd8168"
AUSTEN_ARCHIVE_HASHES = {
    "30ec43e9b6d6815289d65232a076941fba2d2a63cc164ac439f501900bab430e",
    "13a25008072079c700a5afb13ac3a0f1942c60332339d15e013635c7467a9381",
}


def book_records(path: Path):
    """Read only Austen's chapter text, leaving publisher boilerplate out."""
    with tarfile.open(path, "r:gz") as archive:
        chapters = []
        for member in archive:
            match = re.fullmatch(r"jane-austen_pride-and-prejudice-[^/]+/src/epub/text/chapter-(\d+)\.xhtml", member.name)
            if match and member.isfile():
                root = ET.fromstring(archive.extractfile(member).read())
                paragraphs = root.findall(".//{http://www.w3.org/1999/xhtml}p")
                chapters.append((int(match[1]), "\n\n".join("".join(p.itertext()).strip() for p in paragraphs)))
        if len(chapters) != 61:
            raise ValueError("Expected 61 chapters in the Standard Ebooks source")
        yield from sorted(chapters)


def oasst_conversations(path: Path, limit: int) -> list[dict]:
    if sha256(path) != OASST_SHA256:
        raise ValueError("Expected the pinned official OpenAssistant/oasst2 training Parquet")
    parents, roots = defaultdict(list), []
    columns = ["message_id", "parent_id", "text", "role", "lang", "rank", "synthetic", "model_name",
               "review_result", "deleted", "tree_state"]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            if (row["lang"] != "en" or row["synthetic"] is not False or row["model_name"] is not None
                    or row["review_result"] is not True or row["deleted"] is not False
                    or row["tree_state"] != "ready_for_export"):
                continue
            if row["parent_id"] is None and row["role"] == "prompter":
                roots.append(row)
            else:
                parents[row["parent_id"]].append(row)
    roots.sort(key=lambda row: row["message_id"])
    random.Random(42).shuffle(roots)
    conversations, seen = [], set()
    for root in roots:
        messages, node = [], root
        while node is not None:
            messages.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
            children = parents[node["message_id"]]
            if node["role"] == "prompter":
                children = [child for child in children if child["role"] == "assistant" and child["rank"] == 0]
            else:
                children = [child for child in children if child["role"] == "prompter"]
            node = min(children, key=lambda child: child["message_id"], default=None)
        if messages[-1]["role"] == "user":
            messages.pop()
        try:
            messages = clean_messages(messages)
        except ValueError:
            continue
        fingerprint = json.dumps(messages, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        conversations.append({"source": "oasst2-en-human", "id": root["message_id"],
                              "license": "apache-2.0", "origin": "human", "messages": messages})
        if len(conversations) == limit:
            break
    return conversations


def pilot(args):
    book_sha = sha256(args.book)
    if book_sha not in AUSTEN_ARCHIVE_HASHES:
        raise ValueError("Expected the reviewed Standard Ebooks source archive")
    if args.output.exists():
        raise FileExistsError(args.output)
    book_home = "https://github.com/standardebooks/jane-austen_pride-and-prejudice"
    catalog = {
        "tokenizer": {"model": "Qwen/Qwen3-0.6B-Base", "revision": QWEN_REVISION, "sha256": QWEN_TOKENIZER_SHA256},
        "sources": {
            "austen-pride-and-prejudice": {
                "phase": "cpt", "slice": "prose", "license": "public-domain", "origin": "public-domain",
                "source_url": f"{book_home}/tree/{AUSTEN_REVISION}", "revision": AUSTEN_REVISION,
                "artifact_sha256": book_sha, "license_evidence": f"{book_home}/blob/{AUSTEN_REVISION}/LICENSE.md",
                "attribution": "Pride and Prejudice by Jane Austen (1775–1817). Original novel text, first published 1813.",
            },
            "oasst2-en-human": {
                "phase": "sft", "slice": "openassistant", "license": "apache-2.0", "origin": "human",
                "source_url": f"https://huggingface.co/datasets/OpenAssistant/oasst2/tree/{OASST_REVISION}",
                "revision": OASST_REVISION, "artifact_sha256": OASST_SHA256,
                "license_evidence": "https://huggingface.co/datasets/OpenAssistant/oasst2",
                "attribution": "OpenAssistant Conversations 2, LAION/OpenAssistant contributors. English, reviewed export trees and rank-zero human replies.",
            },
        },
    }
    dialogues = oasst_conversations(args.oasst2, args.dialogues + 32)
    if len(dialogues) != args.dialogues + 32:
        raise ValueError(f"Only {len(dialogues)} reviewed conversations matched the filters")
    documents = ({"source": "austen-pride-and-prejudice", "id": f"chapter-{n}",
                  "license": "public-domain", "origin": "public-domain", "text": text}
                 for n, text in book_records(args.book))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".adaptation-pilot-", dir=args.output.parent) as temporary:
        destination = Path(temporary) / "data"
        for phase, split, records, context in [
            ("cpt", "cpt", documents, 8192),
            ("sft", "sft-train", dialogues[32:], args.sft_context),
            ("sft", "sft-eval", dialogues[:32], args.sft_context),
        ]:
            report = build_corpus(records, catalog=catalog, tokenizer_path=args.tokenizer, phase=phase,
                                  destination=destination / split, context=context)
            print(split, sum(shard["rows"] for shard in report["shards"]), "rows", report["rejections"], flush=True)
        destination.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="Pack investigator-reviewed normalized records")
    sample = subparsers.add_parser("pilot", help="Use pinned OASST2 and reviewed Austen inputs")
    for subparser in (pack, sample):
        subparser.add_argument("--tokenizer", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
    pack.add_argument("--catalog", type=Path, required=True)
    pack.add_argument("--records", type=Path, required=True)
    pack.add_argument("--phase", choices=("cpt", "sft"), required=True)
    sample.add_argument("--oasst2", type=Path, required=True)
    sample.add_argument("--book", type=Path, required=True)
    sample.add_argument("--dialogues", type=int, default=96)
    sample.add_argument("--sft-context", type=int, default=512)
    args = parser.parse_args()
    if args.command == "pilot":
        if args.dialogues < 1 or args.sft_context < 2:
            parser.error("Choose a positive sample count and context of at least two")
        pilot(args)
    else:
        with args.records.open(encoding="utf-8") as rows:
            report = build_corpus((json.loads(row) for row in rows if row.strip()),
                                  catalog=json.loads(args.catalog.read_text(encoding="utf-8")),
                                  tokenizer_path=args.tokenizer, phase=args.phase, destination=args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
