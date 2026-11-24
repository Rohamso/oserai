# Connect the Razer laptop to OserAI Chat

The laptop has Windows 11, 16 GB RAM, and an RTX 2070 Max-Q with 8 GB VRAM.
The reported Ollama process already shows Qwen3 4B running entirely on its GPU
with a 4096-token context. This host uses **Ollama**, which lists the RTX 2070
as supported. Parallax's Windows installer lists an RTX 3060 Ti as its minimum;
this setup does not establish Parallax compatibility on the RTX 2070.

- [Ollama hardware support](https://docs.ollama.com/gpu)
- [Parallax Windows requirements](https://github.com/GradientHQ/parallax_win_cli#system-requirements)

## Build and transfer the package

From the OserAI checkout:

```bash
python3 scripts/package_windows_chat.py
```

Copy `out/oserai-chat-windows.zip` to the Razer and extract it into a normal
local folder. The archive contains the gateway source, Windows launcher,
instructions, requirements, and a public `api.php` update. It contains no keys,
private configuration, downloaded binaries, model weights, or usage data.

## Start the host on the Razer

Keep Ollama open, then double-click **Start-Oserai.cmd**. The first launch:

1. Verifies a working 64-bit Python 3.12 or 3.13 interpreter. If none is found,
   checks whether WinGet can run, then installs its official per-user Python
   3.13 package. A manual installer path is provided when WinGet is unavailable.
2. Creates a local virtual environment and installs `requirements-chat.txt`.
3. Reads Ollama's actual installed model tags, including the user-reported
   `qwen3_4b` alias or model digest `359d7dd4bcda` if the standard `qwen3:4b`
   tag is absent. It does not substitute an unrelated model.
4. Creates/reuses a private gateway key in `%LOCALAPPDATA%\OserAIChat` and
   starts the gateway on `127.0.0.1:8010`.
5. Generates a real completed test reply from Ollama through the gateway.
6. Uses an installed `cloudflared` connector or downloads the official Windows
   x64 release from Cloudflare's GitHub release assets. When its asset metadata
   includes a SHA-256 digest, the download must match it.
7. Starts a temporary HTTPS tunnel to the authenticated gateway, verifies its
   status, and creates `%LOCALAPPDATA%\OserAIChat\cpanel-private\chat.php`.

The tunnel uses a separate empty configuration file; existing Cloudflare tunnel
configuration is not modified. Neither Ollama's port nor OserAI's coordinator or
training APIs are exposed. The startup window prints no gateway key.

### If WinGet cannot execute

An error immediately after **Installing the official Python ... package** in
the original launcher occurs at the WinGet command, before the gateway starts.
Use the [official Python 3.13.16 release page](https://www.python.org/downloads/release/python-31316/),
download **Windows installer (64-bit)**, and choose **Install Now**. Close the
failed window and start the updated launcher. It probes the real interpreter
in the usual per-user/all-user locations and through the Python launcher;
WinGet is not needed when a working interpreter is already installed. It also
reports the startup stage when a later command fails.

### If the public HTTPS connection does not become ready

Reaching **Checking the authenticated HTTPS connection** means the local
gateway generated a real reply successfully. The remaining check is the public
tunnel. The launcher prints its public `/health` URL, reports the last HTTP,
DNS, or connection failure, and allows up to three minutes for startup. Local
gateway failures point to `gateway.log`; public tunnel failures point to
`tunnel.log` in `%LOCALAPPDATA%\OserAIChat`.

Read the Cloudflare log in PowerShell:

```powershell
Get-Content "$env:LOCALAPPDATA\OserAIChat\tunnel.log" -Tail 30
```

The launcher explicitly selects automatic protocol fallback. Cloudflare Quick
Tunnels otherwise default to QUIC. If the log reports QUIC/UDP handshake
timeouts, try TCP transport from the extracted package folder:

```powershell
.\Start-Oserai.cmd --tunnel-protocol http2
```

Cloudflare Tunnel needs outbound port 7844 (UDP for QUIC or TCP for HTTP/2).
Forcing HTTP/2 only helps when TCP is available; it cannot fix a DNS failure or
a network that blocks both protocols. A `Registered tunnel connection` log
entry combined with a DNS lookup failure instead points to the public hostname
lookup. Inspect that error before changing the model or website configuration.
[Cloudflare troubleshooting](https://developers.cloudflare.com/tunnel/troubleshooting/)

The host allows one generation at a time, at most 512 output tokens, and 8000
input characters. Ollama uses a 4096-token context, `think: false`, and a 10-minute
keep-alive. Character limits are a request bound, not an exact token count;
Ollama's context window remains the token limit. Rate limits and anonymous usage
storage are the same gateway mechanisms used for Parallax.

## Connect cPanel

When the window says **Connection ready**, use cPanel File Manager for two
uploads:

| File | Destination |
| --- | --- |
| `cpanel-public/api.php` in the extracted ZIP | Replace `api.php` in the exact document root for `chat.oserai.network`, beside `index.html`. |
| The private `chat.php` at the path printed by the launcher | Account **Home → .config → oserai → chat.php**, outside every website document root. |

Enable **Show Hidden Files** in File Manager and create the private folders if
needed. The private file contains the gateway key; upload it directly rather
than copying its contents into chat. It sets `gateway_stream => false` because
[Cloudflare Quick Tunnels do not support SSE](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).
The gateway collects the completed reply as JSON, and PHP returns the browser's
normal chat events after generation. The user sees the reply when it completes.

If PHP does not receive the account's HOME variable, set `OSERAI_PRIVATE_CONFIG`
in the public `api.php` to the actual full private path shown by File Manager,
for example `/home/USERNAME/.config/oserai/chat.php`.

Refresh **https://chat.oserai.network**, wait for the status to become ready,
and send a message. The local test reply verifies Ollama and the gateway; a
successful reply on the website verifies the uploaded PHP configuration too.

## Availability and the next deployment step

Keep the laptop plugged in, Ollama open, and the launcher running. The launcher
prevents automatic system sleep only while it runs. Ctrl+C stops its gateway
and tunnel and restores normal sleep behavior; it leaves Ollama running.

This is a free test connection with no uptime guarantee. Its hostname changes
after restarting, so upload the newly generated **private** `chat.php` each
time. The public PHP update only needs uploading once for this change. After
the first website reply is verified, use a stable HTTPS tunnel or gateway host
for a public MVP; full streaming remains supported when the connection supports
SSE and `gateway_stream` is true.

Automated tests cover Parallax SSE, Ollama NDJSON, JSON completion transport,
private config generation, exact model selection, and the PHP adapter. The
Windows launcher and real RTX 2070 inference must be checked on the Razer; they
cannot be executed on the development Mac. Training and automatic contributor
onboarding remain separate work.
