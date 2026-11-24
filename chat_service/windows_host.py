"""Run the Windows Ollama host and prepare private cPanel connection settings.

No model API, coordinator, or training routes are exposed by the tunnel.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIGEST = "359d7dd4bcda"
ORIGIN = "https://chat.oserai.network"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def json_request(url, *, payload=None, key=None, timeout=15):
    headers = {"Accept": "application/json", "User-Agent": "OserAI-Chat-Host"}
    if key:
        headers.update({"Authorization": f"Bearer {key}",
                        "X-Visitor-ID": hashlib.sha256(b"host-startup-check").hexdigest(),
                        "X-Visitor-Network": hashlib.sha256(b"host-startup-network").hexdigest()})
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
        raw = response.read(500001)
        if len(raw) > 500000:
            raise RuntimeError("Unexpectedly large service response")
        return json.loads(raw)


def choose_model(models, requested=None):
    names = [model.get("name") or model.get("model") for model in models]
    if requested:
        if requested not in names:
            raise RuntimeError("The requested model is not installed in Ollama: " + requested)
        return requested
    for preferred in ("qwen3:4b", "qwen3_4b:latest", "qwen3_4b"):
        if preferred in names:
            return preferred
    for model in models:
        if str(model.get("digest", "")).removeprefix("sha256:").startswith(MODEL_DIGEST):
            name = model.get("name") or model.get("model")
            if name:
                return name
    raise RuntimeError("Qwen3 4B was not found. In PowerShell run: ollama pull qwen3:4b")


def write_private(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    if os.name != "nt":
        path.chmod(0o600)


def load_key(state: Path) -> str:
    path = state / "gateway-key.txt"
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{64}", key):
            raise RuntimeError("Invalid private gateway key file: " + str(path))
        return key
    key = secrets.token_urlsafe(48)
    write_private(path, key + "\n")
    return key


def gateway_environment(state: Path, key: str, model: str):
    values = {
        "GATEWAY_KEY": key, "INFERENCE_BACKEND": "ollama",
        "OLLAMA_URL": "http://127.0.0.1:11434", "OLLAMA_CONTEXT_LENGTH": "4096",
        "MODEL_NAME": model, "MODEL_LABEL": "OserAI Chat",
        "DB_PATH": str(state / "usage.db"), "MAX_CONCURRENT": "1",
        "MAX_INPUT_CHARS": "8000", "MAX_OUTPUT_TOKENS": "512", "TIMEOUT_S": "90",
        "REQUESTS_PER_MINUTE": "6", "REQUESTS_PER_DAY": "100",
    }
    environment = os.environ.copy()
    environment.update({"OSERAI_CHAT_" + name: value for name, value in values.items()})
    return environment


def write_cpanel_config(state: Path, url: str, key: str):
    if not TUNNEL_URL.fullmatch(url) or not re.fullmatch(r"[A-Za-z0-9_-]{64}", key):
        raise ValueError("Invalid gateway configuration")
    path = state / "cpanel-private" / "chat.php"
    write_private(path, "<?php\nreturn [\n"
                  f"    'gateway_url' => '{url}',\n"
                  f"    'gateway_key' => '{key}',\n"
                  f"    'allowed_origin' => '{ORIGIN}',\n"
                  "    'allow_local_http' => false,\n"
                  "    'gateway_stream' => false,\n];\n")
    return path


def cloudflared_binary(state: Path):
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    target = state / "bin" / "cloudflared.exe"
    if target.exists():
        return str(target)
    print("Downloading Cloudflare's Windows tunnel connector...")
    release = json_request("https://api.github.com/repos/cloudflare/cloudflared/releases/latest", timeout=30)
    asset = next((item for item in release["assets"] if item["name"] == "cloudflared-windows-amd64.exe"), None)
    if asset is None or not asset["browser_download_url"].startswith("https://github.com/cloudflare/cloudflared/releases/download/"):
        raise RuntimeError("The official Windows tunnel download could not be found")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as source, temporary.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > 200 * 1024 * 1024:
                    raise RuntimeError("Unexpected tunnel download size")
                digest.update(chunk)
                output.write(chunk)
        expected = asset.get("digest")
        if expected and expected != "sha256:" + digest.hexdigest():
            raise RuntimeError("The tunnel connector checksum did not match")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


def wait_until_ready(url, process, key=None, *, stage="Local chat gateway", timeout_s=60, log_path=None):
    deadline = time.monotonic() + timeout_s
    last_problem = "No response received"
    reported = None
    expected = "ready" if key else "ok"
    log_note = f" See {log_path}." if log_path else ""
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"{stage} stopped (exit code {exit_code})." + log_note)
        try:
            result = json_request(url, key=key, timeout=5)
            if isinstance(result, dict) and result.get(expected) is True:
                return
            last_problem = f"The endpoint responded but did not report {expected}=true"
        except urllib.error.HTTPError as error:
            last_problem = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            last_problem = f"Connection error: {error.reason}"
        except OSError as error:
            last_problem = f"Connection error: {error}"
        except ValueError:
            last_problem = "The endpoint did not return valid JSON"
        # Errors can contain a configured proxy URL; never print credentials.
        last_problem = re.sub(r"(https?://)[^\s/@]+@", r"\1[redacted]@", last_problem)
        if key:
            last_problem = last_problem.replace(key, "[redacted]")
        last_problem = last_problem[:500]
        if last_problem != reported:
            print(f"{stage}: {last_problem}. Waiting for connection...", flush=True)
            reported = last_problem
        time.sleep(1)
    raise RuntimeError(f"{stage} did not become ready at {url}. Last check: {last_problem}." + log_note)


def capture_tunnel(process, state, lines):
    with (state / "tunnel.log").open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            try:
                lines.put_nowait(line)
            except queue.Full:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact Ollama model name; otherwise detect the reported Qwen3 4B model")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--tunnel-protocol", choices=("auto", "http2", "quic"), default="auto",
                        help="Cloudflare transport; use http2 when the tunnel log reports QUIC/UDP timeouts")
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This launcher is for the Windows laptop. Run it there after extracting the ZIP.")
    state = args.state_dir or Path(os.environ["LOCALAPPDATA"]) / "OserAIChat"
    state.mkdir(parents=True, exist_ok=True)
    print("OserAI chat host - keep this window open while serving chats.")
    print("Checking the local Ollama model...")
    inventory = json_request("http://127.0.0.1:11434/api/tags")
    model = choose_model(inventory["models"], args.model)
    print("Using installed model: " + model)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            raise RuntimeError(f"Port {args.port} is already in use. Close the other OserAI host window first.") from None
    key = load_key(state)
    connector = cloudflared_binary(state)
    processes = []
    gateway_log = (state / "gateway.log").open("w", encoding="utf-8")
    try:
        # Prevent automatic sleep only while this launcher is running.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        gateway = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "chat_service.main:app",
            "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1", "--no-access-log",
        ], cwd=ROOT, env=gateway_environment(state, key, model), stdout=gateway_log, stderr=subprocess.STDOUT)
        processes.append(gateway)
        local_url = f"http://127.0.0.1:{args.port}"
        print("Checking the local chat gateway...", flush=True)
        wait_until_ready(local_url + "/health", gateway, log_path=state / "gateway.log")
        print("Generating a real test reply on your laptop...")
        reply = json_request(local_url + "/api/chat", key=key, timeout=100,
                             payload={"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "stream": False})
        if not isinstance(reply.get("text"), str) or not reply["text"].strip():
            raise RuntimeError("The model did not return a completed test reply")
        print("Model reply: " + reply["text"][:300])
        empty_config = state / "tunnel-empty.yml"
        empty_config.write_text("{}\n", encoding="utf-8")
        tunnel = subprocess.Popen([
            connector, "tunnel", "--config", str(empty_config), "--url", local_url,
            "--no-autoupdate", "--loglevel", "info", "--metrics", "127.0.0.1:0",
            "--protocol", args.tunnel_protocol,
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        processes.append(tunnel)
        lines = queue.Queue(maxsize=200)
        threading.Thread(target=capture_tunnel, args=(tunnel, state, lines), daemon=True).start()
        deadline = time.monotonic() + 90
        public_url = None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError("The temporary tunnel stopped; see tunnel.log in the private state folder")
            try:
                match = TUNNEL_URL.search(lines.get(timeout=0.5))
                if match:
                    public_url = match.group(0)
                    break
            except queue.Empty:
                pass
        if not public_url:
            raise RuntimeError("The temporary tunnel did not provide a hostname; see tunnel.log")
        print("Checking the authenticated HTTPS connection...", flush=True)
        print("Public health-check URL: " + public_url + "/health", flush=True)
        wait_until_ready(public_url + "/api/status", tunnel, key=key, stage="Public HTTPS tunnel",
                         timeout_s=180, log_path=state / "tunnel.log")
        config_path = write_cpanel_config(state, public_url, key)
        print("\nConnection ready. Complete these two uploads in cPanel File Manager:")
        print("1. Upload cpanel-public\\api.php from the extracted ZIP to the chat website's document root.")
        print("2. Upload this private file: " + str(config_path))
        print("   Destination: Home / .config / oserai / chat.php (outside the website document root).")
        print("   Enable 'Show Hidden Files' and create those folders if needed.")
        print("\nThen open " + ORIGIN + " and send a message.")
        print("This test tunnel changes its URL after restarting. Upload the newly generated private chat.php each time.")
        print("Keep Ollama and this window running, with the laptop plugged in. Ctrl+C stops the host.")
        print("Private logs and configuration: " + str(state))
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError("The gateway or tunnel stopped. Restart this launcher and upload the new private chat.php.")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        gateway_log.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOserAI chat host stopped.")
    except Exception as error:
        print("\nCould not start OserAI: " + str(error), file=sys.stderr)
        sys.exit(1)
