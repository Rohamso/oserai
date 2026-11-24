"""Package the Windows chat host and public PHP update, excluding runtime state."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Start-Oserai.cmd": "deploy/windows-chat/Start-Oserai.cmd",
    "README.txt": "deploy/windows-chat/README.txt",
    "requirements-chat.txt": "requirements-chat.txt",
    "chat_service/__init__.py": "chat_service/__init__.py",
    "chat_service/config.py": "chat_service/config.py",
    "chat_service/inference.py": "chat_service/inference.py",
    "chat_service/knowledge.py": "chat_service/knowledge.py",
    "chat_service/main.py": "chat_service/main.py",
    "chat_service/pilot.py": "chat_service/pilot.py",
    "chat_service/usage.py": "chat_service/usage.py",
    "chat_service/windows_host.py": "chat_service/windows_host.py",
    "docs/parallax-mvp.md": "docs/parallax-mvp.md",
    "desktop_app/README.md": "desktop_app/README.md",
    "cpanel-public/api.php": "web/chat/public/api.php",
}


def package(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, source in FILES.items():
            path = ROOT / source
            if not path.is_file() or path.is_symlink():
                raise ValueError("Missing package file or unexpected symlink: " + source)
            data = path.read_bytes()
            if name.endswith((".cmd", ".txt")):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            archive.writestr(name, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/oserai-chat-windows.zip"))
    args = parser.parse_args()
    package(args.output)
    print(f"Windows chat host: {args.output.resolve()} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
