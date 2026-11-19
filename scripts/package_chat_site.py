"""Build the upload-ready cPanel ZIP from an explicit public-file allowlist."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "web" / "chat" / "public"
FILES = (
    "index.html", "api.php", ".htaccess", ".user.ini", "assets/app.js",
    "assets/app.css", "assets/icons.svg", "assets/favicon.svg",
)
FRONTEND_FILES = tuple(name for name in FILES if name not in ("api.php", ".user.ini"))


def package(output: Path, *, frontend_only: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = FRONTEND_FILES if frontend_only else FILES
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = PUBLIC / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing public asset or unexpected symlink: {name}")
            entry = zipfile.ZipInfo.from_file(path, name)
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frontend-only", action="store_true", help="Include the HTML, styles, scripts, and icons.")
    args = parser.parse_args()
    default = "out/oserai-chat-frontend.zip" if args.frontend_only else "out/oserai-chat-cpanel.zip"
    output = args.output or Path(default)
    package(output, frontend_only=args.frontend_only)
    print(f"cPanel upload: {output.resolve()} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
