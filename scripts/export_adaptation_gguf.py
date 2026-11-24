"""Merge a local adaptation pilot adapter and export a Q8_0 model for Colibri."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converter = args.llama_cpp.resolve() / "convert_hf_to_gguf.py"
    if not converter.is_file() or args.output.exists():
        parser.error("Point to a llama.cpp checkout and choose an unused GGUF output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with TemporaryDirectory(prefix="adaptation-export-", dir=args.output.parent) as temporary:
        workspace = Path(temporary)
        merged = workspace / "merged"
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
            "message['content'] + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        tokenizer.save_pretrained(merged)
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16, local_files_only=True)
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload(safe_merge=True)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.save_pretrained(merged)
        del model
        gguf = workspace / "model.gguf"
        subprocess.run([sys.executable, str(converter), str(merged), "--outtype", "q8_0",
                        "--outfile", str(gguf)], check=True)
        args.output.hardlink_to(gguf)
    print(args.output)


if __name__ == "__main__":
    main()
