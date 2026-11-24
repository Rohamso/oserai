"""Audit, filter, and pack reviewed inputs for native-tokenizer CPT and ChatML SFT."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash
from tokenizers import Tokenizer


LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "public-domain"}
ORIGINS = {"human", "public-domain", "verified-open-model", "publisher-generated"}
CODE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}
CODE_LANGUAGES = {"python", "javascript", "typescript", "shell", "sql"}
SYSTEM_PROMPT = "You are an everyday AI assistant operating on a decentralized network."
SEPARATOR, TURN_START, TURN_END = "<|endoftext|>", "<|im_start|>", "<|im_end|>"
MASK = -100
PII = re.compile(
    r"(?P<email>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<ssn>\b\d{3}-\d{2}-\d{4}\b)"
    r"|(?P<phone>(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}\b)"
)
CONTROL_MARKER = re.compile(r"<\|[^|\n]+\|>")
WORDS = re.compile(r"\w+", re.UNICODE)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def clean_text(text: str, minimum_words: int) -> str:
    if not isinstance(text, str) or CONTROL_MARKER.search(text):
        raise ValueError("invalid_text")
    text = text.strip()
    words = len(text.split())
    if words < minimum_words:
        raise ValueError("too_short")
    # Redact low-density matches. Reject records with at least three known
    # identifiers and a density of three or more per 100 words.
    text, count = PII.subn("[redacted]", text)
    if count >= 3 and count * 100 >= 3 * words:
        raise ValueError("pii_density")
    return text


def clean_messages(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError("invalid_dialogue")
    result = []
    for index, message in enumerate(messages):
        role = "assistant" if index % 2 else "user"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("invalid_dialogue")
        result.append({"role": role, "content": clean_text(message.get("content"), 1)})
    return result


def chatml_tokens(tokenizer: Tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    text = f"{TURN_START}system\n{SYSTEM_PROMPT}{TURN_END}\n"
    spans = []
    for message in messages:
        role = message["role"]
        text += f"{TURN_START}{role}\n"
        start = len(text)
        text += message["content"] + TURN_END
        if role == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    encoding = tokenizer.encode(text, add_special_tokens=False)
    labels = [token if any(start <= low < high <= end for start, end in spans) else MASK
              for token, (low, high) in zip(encoding.ids, encoding.offsets, strict=True)]
    return encoding.ids, labels


class DedupIndex:
    """Keep exact and MinHash LSH indexes on disk across inputs and shards."""

    def __init__(self, database: sqlite3.Connection):
        self.db = database
        self.db.executescript("""
            CREATE TABLE documents (id INTEGER PRIMARY KEY, exact BLOB UNIQUE, signature BLOB);
            CREATE TABLE bands (bucket BLOB NOT NULL, doc_id INTEGER NOT NULL);
            CREATE INDEX band_bucket ON bands(bucket);
        """)

    def add(self, text: str, *, near: bool) -> bool:
        words = WORDS.findall(text.casefold())
        normalized = " ".join(words) if near else text
        exact = hashlib.sha256(normalized.encode()).digest()
        if self.db.execute("SELECT 1 FROM documents WHERE exact=?", (exact,)).fetchone():
            return False
        signature, buckets = b"", []
        if near:
            minhash = MinHash(num_perm=64, seed=42)
            shingles = (" ".join(words[index:index + 5]).encode() for index in range(max(1, len(words) - 4)))
            while batch := list(islice(shingles, 2048)):
                minhash.update_batch(batch)
            signature = minhash.hashvalues.astype("<u8").tobytes()
            bands = [signature[offset:offset + 64] for offset in range(0, len(signature), 64)]
            buckets = [bytes([band]) + value for band, value in enumerate(bands)]
            candidates = self.db.execute(
                "SELECT DISTINCT signature FROM documents WHERE id IN "
                "(SELECT doc_id FROM bands WHERE bucket IN (?,?,?,?,?,?,?,?))", buckets,
            )
            values = memoryview(signature).cast("Q")
            for (candidate,) in candidates:
                equal = sum(a == b for a, b in zip(values, memoryview(candidate).cast("Q"), strict=True))
                if equal / 64 >= 0.8:
                    return False
        cursor = self.db.execute("INSERT INTO documents(exact,signature) VALUES(?,?)", (exact, signature))
        self.db.executemany("INSERT INTO bands VALUES(?,?)", ((bucket, cursor.lastrowid) for bucket in buckets))
        return True


REFERENCE = pa.struct([
    ("source", pa.string()), ("record_id", pa.string()), ("row_offset", pa.int32()),
    ("document_offset", pa.int64()), ("length", pa.int32()),
])


class Shards:
    def __init__(self, root: Path, phase: str, context: int, max_bytes: int):
        fields = [("input_ids", pa.list_(pa.uint32(), context) if phase == "cpt" else pa.list_(pa.uint32())),
                  ("references", pa.list_(REFERENCE))]
        if phase == "sft":
            fields.append(("labels", pa.list_(pa.int32())))
        self.schema = pa.schema(fields)
        self.root, self.phase, self.max_bytes = root, phase, max_bytes
        self.buffer, self.manifest = [], []
        self.path = self.sink = self.writer = None
        self.rows = self.tokens = 0
        self.sources = set()

    def append(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) == 128:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if self.writer is None:
            self.path = self.root / f"{self.phase}-{len(self.manifest):05d}.parquet"
            self.sink = self.path.open("wb")
            self.writer = pq.ParquetWriter(self.sink, self.schema, compression="NONE", use_dictionary=False)
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        self.rows += len(self.buffer)
        self.tokens += sum(len(row["input_ids"]) for row in self.buffer)
        self.sources.update(ref["source"] for row in self.buffer for ref in row["references"])
        self.buffer.clear()
        if self.sink.tell() >= self.max_bytes:
            self.close_shard()

    def close_shard(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.sink.close()
        self.manifest.append({"file": self.path.name, "sha256": sha256(self.path), "bytes": self.path.stat().st_size,
                              "rows": self.rows, "tokens": self.tokens, "sources": sorted(self.sources)})
        self.path = self.writer = self.sink = None
        self.rows = self.tokens = 0
        self.sources.clear()


def validate_catalog(catalog: dict, phase: str, tokenizer_path: Path) -> None:
    identity = catalog["tokenizer"]
    if not identity.get("model") or not identity.get("revision") or sha256(tokenizer_path) != identity.get("sha256"):
        raise ValueError("Tokenizer does not match the reviewed model, revision, and SHA-256")
    for source, entry in catalog["sources"].items():
        if entry["phase"] != phase:
            continue
        if (entry.get("license") not in LICENSES or entry.get("origin") not in ORIGINS
                or not all(entry.get(field) for field in ("revision", "source_url", "license_evidence", "artifact_sha256"))):
            raise ValueError(f"Source lacks reviewed, allowlisted provenance: {source}")


def build_corpus(records: Iterable[dict], *, catalog: dict, tokenizer_path: Path, phase: str,
                 destination: Path, context: int = 8192, shard_bytes: int = 1_500_000_000) -> dict:
    """Publish a new corpus atomically. A CPT tail shorter than one row is recorded and dropped."""
    if phase not in {"cpt", "sft"} or context < 2 or shard_bytes < 1:
        raise ValueError("Invalid phase, context, or shard size")
    if destination.exists():
        raise FileExistsError(destination)
    validate_catalog(catalog, phase, tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for marker in (SEPARATOR, TURN_START, TURN_END):
        tokens = tokenizer.encode(marker, add_special_tokens=False).ids
        if len(tokens) != 1 or tokens[0] != tokenizer.token_to_id(marker):
            raise ValueError(f"Native tokenizer is missing atomic marker {marker}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    accepted = Counter()
    rejected = Counter()
    packed, references = [], []
    with TemporaryDirectory(prefix=".adaptation-", dir=destination.parent) as temporary:
        work = Path(temporary)
        output = work / "corpus"
        output.mkdir()
        shards = Shards(output, phase, context, shard_bytes)
        connection = sqlite3.connect(work / "dedup.sqlite")
        try:
            dedup = DedupIndex(connection)
            for index, record in enumerate(records):
                if index % 1000 == 0:
                    connection.commit()
                if not isinstance(record, dict) or not isinstance(record.get("source"), str):
                    rejected["provenance"] += 1
                    continue
                key, record_id = record["source"], record.get("id")
                entry = catalog["sources"].get(key)
                if (not entry or entry["phase"] != phase or record.get("license") != entry["license"]
                        or record.get("origin") != entry["origin"] or not isinstance(record_id, str) or not record_id):
                    rejected["provenance"] += 1
                    continue
                try:
                    if phase == "cpt":
                        category = entry.get("slice")
                        if category == "everyday_web":
                            score = record.get("education_score")
                            if type(score) not in (int, float) or not 3 <= score < float("inf"):
                                raise ValueError("education_score")
                        if category == "logic_code" and (entry["license"] not in CODE_LICENSES
                                or record.get("language") not in CODE_LANGUAGES):
                            raise ValueError("code_license_or_language")
                        if category == "stem_health" and entry["license"] not in {"cc0-1.0", "cc-by-4.0"}:
                            raise ValueError("stem_license")
                        text = clean_text(record.get("text"), 50)
                        tokens = tokenizer.encode(text, add_special_tokens=False).ids + [tokenizer.token_to_id(SEPARATOR)]
                    else:
                        messages = clean_messages(record.get("messages"))
                        text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                        tokens, labels = chatml_tokens(tokenizer, messages)
                        if len(tokens) > context:
                            raise ValueError("over_context")
                except ValueError as error:
                    rejected[str(error)] += 1
                    continue
                if not dedup.add(text, near=phase == "cpt"):
                    rejected["duplicate"] += 1
                    continue
                accepted[key] += 1
                if phase == "sft":
                    shards.append({"input_ids": tokens, "labels": labels, "references": [
                        {"source": key, "record_id": record_id, "row_offset": 0, "document_offset": 0, "length": len(tokens)},
                    ]})
                    continue
                offset = 0
                while offset < len(tokens):
                    count = min(context - len(packed), len(tokens) - offset)
                    references.append({"source": key, "record_id": record_id, "row_offset": len(packed),
                                       "document_offset": offset, "length": count})
                    packed.extend(tokens[offset:offset + count])
                    offset += count
                    if len(packed) == context:
                        shards.append({"input_ids": packed, "references": references})
                        packed, references = [], []
            shards.flush()
        finally:
            connection.close()
            shards.close_shard()
        if not shards.manifest:
            raise ValueError("No eligible rows were written")
        manifest = {
            "schema_version": 1, "phase": phase, "context": context, "compression": "uncompressed",
            "tokenizer": catalog["tokenizer"], "sources": {key: catalog["sources"][key] for key in sorted(accepted)},
            "accepted_records": dict(accepted), "rejections": dict(rejected), "dropped_tail_tokens": len(packed),
            "shards": shards.manifest,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        output.rename(destination)
    return manifest
