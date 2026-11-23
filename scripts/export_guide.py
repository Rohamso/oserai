"""Merge OserAI Guide's PEFT adapter and export a Q8 GGUF for Colibri."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ADAPTER = Path(__file__).resolve().parents[1] / "models" / "oserai-guide-v0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, required=True, help="Path to the llama.cpp checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converter = args.llama_cpp.resolve() / "convert_hf_to_gguf.py"
    if not converter.is_file():
        parser.error("--llama-cpp must point to a checkout containing convert_hf_to_gguf.py")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose an output file that does not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = PeftConfig.from_pretrained(ADAPTER)
    options = {"revision": config.revision, "trust_remote_code": False}
    torch.set_num_threads(4)
    with TemporaryDirectory(prefix="guide-export-", dir=output.parent) as staging:
        staging = Path(staging)
        merged = staging / "merged"
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=torch.float16, **options)
        model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True)
        del model
        AutoTokenizer.from_pretrained(config.base_model_name_or_path, **options).save_pretrained(merged)
        gguf = staging / "guide.gguf"
        subprocess.run([sys.executable, str(converter), str(merged), "--outtype", "q8_0",
                        "--outfile", str(gguf)], check=True)
        # A hard link publishes the verified converter output without replacing an existing file.
        output.hardlink_to(gguf)
    print(output)


if __name__ == "__main__":
    main()
