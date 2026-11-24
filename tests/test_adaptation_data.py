from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_DATA_DEPS = all(importlib.util.find_spec(name) for name in ("pyarrow", "datasketch", "tokenizers"))
if HAS_DATA_DEPS:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers

    from training.adaptation_data import MASK, build_corpus, chatml_tokens, clean_text, sha256


@unittest.skipUnless(HAS_DATA_DEPS, "Install requirements-adaptation-data.txt")
class TestAdaptationData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        vocab = {"[UNK]": 0, "<|endoftext|>": 1, "<|im_start|>": 2, "<|im_end|>": 3, "answer": 4, "test": 5}
        self.tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer.save(str(self.tokenizer_path))
        self.source = {"phase": "cpt", "slice": "prose", "license": "cc0-1.0", "origin": "human",
                       "revision": "fixture-revision", "source_url": "https://example.com/source",
                       "license_evidence": "https://example.com/license", "artifact_sha256": "a" * 64}
        self.catalog = {"tokenizer": {"model": "test-model", "revision": "pinned",
                                      "sha256": sha256(self.tokenizer_path)}, "sources": {"fixture": self.source}}

    def build(self, records):
        return build_corpus(records, catalog=self.catalog, tokenizer_path=self.tokenizer_path,
                            phase="cpt", destination=self.root / "packed", context=32)

    def test_chatml_masks_prompt_and_keeps_assistant_content_and_stop(self):
        ids, labels = chatml_tokens(self.tokenizer, [
            {"role": "user", "content": "test"}, {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "test"}, {"role": "assistant", "content": "answer"},
        ])
        self.assertEqual([value for value in labels if value != MASK], [4, 3, 4, 3])
        self.assertTrue(all(label == MASK or label == token for token, label in zip(ids, labels)))

    def test_rejects_share_alike_catalog_without_publishing_output(self):
        self.source["license"] = "cc-by-sa-3.0"
        with self.assertRaisesRegex(ValueError, "allowlisted provenance"):
            self.build([])
        self.assertFalse((self.root / "packed").exists())

    def test_cpt_packs_deduplicated_records_with_offsets_and_hash(self):
        first = {"source": "fixture", "id": "doc-a", "license": "cc0-1.0", "origin": "human",
                 "text": "test " * 62 + "answer"}
        report = self.build([first, {**first, "id": "dupe"}, {**first, "license": "cc-by-sa-3.0"}])
        shard = report["shards"][0]
        self.assertEqual(report["rejections"], {"duplicate": 1, "provenance": 1})
        self.assertEqual((shard["rows"], shard["tokens"]), (2, 64))
        path = self.root / "packed" / shard["file"]
        self.assertEqual(shard["sha256"], sha256(path))
        rows = pq.read_table(path).to_pylist()
        self.assertEqual(rows[1]["input_ids"][-1], 1)
        self.assertEqual(rows[1]["references"][0]["document_offset"], 32)
        self.assertEqual(clean_text("Reach me at test@example.com today", 1), "Reach me at [redacted] today")
