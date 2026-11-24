from __future__ import annotations

import os
import socket
import urllib.error
import zipfile
from types import SimpleNamespace

import pytest

from chat_service.windows_host import choose_model, gateway_environment, load_key, write_cpanel_config
from chat_service import windows_host
from scripts.package_windows_chat import FILES, package


def test_model_discovery_uses_exact_tags_and_reported_digest():
    assert choose_model([{"name": "qwen3:4b"}]) == "qwen3:4b"
    assert choose_model([{"name": "qwen3_4b:latest"}]) == "qwen3_4b:latest"
    assert choose_model([{"name": "my-chat:latest", "digest": "sha256:359d7dd4bcda1234"}]) == "my-chat:latest"
    assert choose_model([{"name": "custom:latest"}], "custom:latest") == "custom:latest"
    with pytest.raises(RuntimeError, match="not found"):
        choose_model([{"name": "unrelated-model:latest"}])
    with pytest.raises(RuntimeError, match="not installed"):
        choose_model([{"name": "qwen3:4b"}], "qwen3_4b")


def test_host_reuses_private_key_and_does_not_export_it_to_the_parent(tmp_path):
    key = load_key(tmp_path)
    assert len(key) == 64 and load_key(tmp_path) == key
    original = os.environ.get("OSERAI_CHAT_GATEWAY_KEY")
    environment = gateway_environment(tmp_path, key, "my-chat:latest")
    assert environment["OSERAI_CHAT_GATEWAY_KEY"] == key
    assert environment["OSERAI_CHAT_MODEL_NAME"] == "my-chat:latest"
    assert environment["OSERAI_CHAT_MAX_CONCURRENT"] == "1"
    assert os.environ.get("OSERAI_CHAT_GATEWAY_KEY") == original
    path = write_cpanel_config(tmp_path, "https://test-oserai.trycloudflare.com", key)
    text = path.read_text()
    assert "'gateway_stream' => false" in text
    assert "'allowed_origin' => 'https://chat.oserai.network'" in text
    assert key in text
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        write_cpanel_config(tmp_path, "https://wrong.example/'injected", key)


def test_host_rejects_a_corrupted_key_instead_of_changing_it(tmp_path):
    (tmp_path / "gateway-key.txt").write_text("broken-key")
    with pytest.raises(RuntimeError, match="Invalid private"):
        load_key(tmp_path)


def test_windows_archive_contains_only_the_explicit_sources(tmp_path):
    path = tmp_path / "host.zip"
    package(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(FILES)
        assert not any("gateway-key" in name or "cpanel-private" in name or ".venv" in name for name in archive.namelist())
        assert "OSERAI_PRIVATE_CONFIG" in archive.read("cpanel-public/api.php").decode()
        assert "parallax join" in archive.read("docs/parallax-mvp.md").decode()
        assert "What a contributor needs to fill in" in archive.read("desktop_app/README.md").decode()
        launcher = archive.read("Start-Oserai.cmd")
        assert b"\r\n" in launcher and b"\n" not in launcher.replace(b"\r\n", b"")


def fake_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(windows_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(windows_host.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))


def test_public_readiness_retries_temporary_http_errors_until_model_is_available(monkeypatch, capsys):
    fake_clock(monkeypatch)
    results = iter([urllib.error.HTTPError("https://test.trycloudflare.com", 502, "Bad Gateway", {}, None),
                    {"ok": True}, {"ready": True}])

    def fetch(*args, **kwargs):
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(windows_host, "json_request", fetch)
    windows_host.wait_until_ready("https://test.trycloudflare.com/api/status", SimpleNamespace(poll=lambda: None),
                                 key="test-only-key", stage="Public HTTPS tunnel", timeout_s=5)
    output = capsys.readouterr().out
    assert "HTTP 502" in output
    assert "ready=true" in output


def test_public_readiness_keeps_the_dns_failure_and_correct_log_path(monkeypatch, tmp_path):
    fake_clock(monkeypatch)

    def fetch(*args, **kwargs):
        raise urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))

    monkeypatch.setattr(windows_host, "json_request", fetch)
    with pytest.raises(RuntimeError, match="Public HTTPS tunnel.*getaddrinfo failed") as error:
        windows_host.wait_until_ready("https://test.trycloudflare.com/api/status", SimpleNamespace(poll=lambda: None),
                                     key="test-only-key", stage="Public HTTPS tunnel", timeout_s=2,
                                     log_path=tmp_path / "tunnel.log")
    assert "tunnel.log" in str(error.value) and "gateway.log" not in str(error.value)


def test_readiness_diagnostics_hide_keys_and_proxy_credentials(monkeypatch, capsys):
    fake_clock(monkeypatch)

    def fetch(*args, **kwargs):
        raise urllib.error.URLError("proxy http://private-user:private-password@proxy.example failed secret-key")

    monkeypatch.setattr(windows_host, "json_request", fetch)
    with pytest.raises(RuntimeError) as error:
        windows_host.wait_until_ready("https://test.trycloudflare.com/api/status", SimpleNamespace(poll=lambda: None),
                                     key="secret-key", stage="Public HTTPS tunnel", timeout_s=2)
    output = str(error.value) + capsys.readouterr().out
    assert "private-user" not in output and "private-password" not in output and "secret-key" not in output


def test_readiness_reports_a_stopped_process_without_another_request(monkeypatch):
    monkeypatch.setattr(windows_host, "json_request", lambda *args, **kwargs: pytest.fail("Stopped process should not be probed"))
    with pytest.raises(RuntimeError, match="Public HTTPS tunnel stopped.*exit code 1.*tunnel.log"):
        windows_host.wait_until_ready("https://test.trycloudflare.com/api/status", SimpleNamespace(poll=lambda: 1),
                                     stage="Public HTTPS tunnel", log_path="tunnel.log")
