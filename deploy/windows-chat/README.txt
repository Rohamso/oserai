OserAI chat host for the Razer laptop (Windows 11, RTX 2070 8 GB)

1. Copy the ZIP to the Razer. Extract the whole ZIP into a normal local folder.
   Keep Ollama running with the Qwen3 4B model already downloaded.
2. Double-click Start-Oserai.cmd. First startup finds Python 3.12 or 3.13 and
   installs Python 3.13 through WinGet if needed and available, followed by
   the chat-service Python dependencies, and Cloudflare's Windows connector.
   It selects the exact installed model name and generates a real test reply.
3. When the window says "Connection ready", upload these files using cPanel:

   PUBLIC FILE: cpanel-public/api.php from this package
   Destination: the document root for chat.oserai.network, beside index.html.
   Replace the existing api.php. Keep the rest of the website as it is.

   PRIVATE FILE: the chat.php file at the exact path printed by the launcher
   Destination: cPanel Home / .config / oserai / chat.php.
   In File Manager settings, enable "Show Hidden Files". Create .config and
   oserai inside your account's Home folder if needed.
   Do not put this private file in public_html or any website document root.
   It contains the gateway key; do not paste its contents into chat.

4. Refresh https://chat.oserai.network and send a message.

Keep Ollama and the launcher window running with the laptop plugged in. The
launcher prevents automatic sleep while it is running and restores that setting
when stopped. Ctrl+C stops the gateway and tunnel without stopping Ollama.

This is a temporary, free test connection. Cloudflare Quick Tunnels do not
support streaming replies, so each complete reply appears after generation.
The tunnel URL changes when restarted: upload the newly generated PRIVATE
chat.php again. You only need to upload the PUBLIC api.php once for this update.
A stable tunnel/hostname is the next step after verifying the first real chat.

The gateway accepts one generation at a time, uses a 4096-token model context,
and limits output to 512 tokens. It disables thinking for Qwen3 and keeps the
model loaded for 10 minutes after a request. The host stores anonymous usage
counts, not message text. Gateway keys, logs, and usage stay in this Windows
user's AppData/Local/OserAIChat folder, outside the extracted public files.

No router port forwarding, inbound firewall rule, or public Ollama endpoint is
needed. The HTTPS tunnel connects to the authenticated OserAI gateway only.
This host uses Ollama; it does not validate Parallax on the RTX 2070 and does
not train a model. OserAI's Parallax and training integrations remain separate.

Troubleshooting:
- If the old launcher failed after "Installing the official Python ...", WinGet
  could not start or install Python. Download the normal Windows installer
  (64-bit) from https://www.python.org/downloads/release/python-31316/ and choose
  Install Now. Close the failed window, then run this updated launcher again.
  It detects the actual installed Python executable without requiring WinGet.
- "OserAI stopped at" names the failed stage; share it with the preceding
  error lines. The launcher checks Python execution instead of file existence.
- "Qwen3 4B was not found": run ollama pull qwen3:4b, then start again.
- "Connection refused" checking Ollama: open the Ollama app, then start again.
- Port 8010 in use: close the earlier OserAI host window before restarting.
- If startup reaches "Checking the authenticated HTTPS connection", the local
  gateway and model test have passed. The launcher now prints the actual
  public-check failure (HTTP status, DNS/connection error, or invalid response)
  and allows up to three minutes for the tunnel and hostname to become ready.
  Read tunnel.log in the state folder for Cloudflare connection errors.
- If tunnel.log reports repeated QUIC or UDP handshake timeouts, open PowerShell
  in this extracted folder and run:
    .\Start-Oserai.cmd --tunnel-protocol http2
  This selects TCP transport. If TCP port 7844 is also blocked, the connection
  needs a network that permits Cloudflare Tunnel traffic; protocol selection
  alone cannot fix that. The default explicitly selects automatic fallback.
- The printed public /health URL is safe to open in a browser while the host
  is running. A JSON response containing "ok":true confirms that connection;
  a tunnel that has stopped will no longer answer on its old URL.
- The site stays "coming online": check that both uploads went to their exact
  destinations, and that chat.php was regenerated after the latest restart.
  If PHP does not receive your account's HOME variable, set OSERAI_PRIVATE_CONFIG
  in the public api.php to the full private /home/USERNAME/.config/oserai/chat.php
  path shown by cPanel File Manager.
- The state folder printed by the launcher has gateway.log and tunnel.log for
  startup problems. Share the error text, never gateway-key.txt or chat.php.

Sources:
https://docs.ollama.com/gpu
https://docs.ollama.com/api/chat
https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
