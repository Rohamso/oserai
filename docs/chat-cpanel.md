# Publish OserAI Chat at chat.oserai.network

The website is ready to upload to Namecheap cPanel. The uploaded site will
show **“Chat is coming online”** until the inference
service is connected. There are no simulated answers in the website.

For the Windows 11 Razer laptop using its installed Ollama model, follow the
[Windows host guide](chat-windows.md). The steps below describe the Parallax
deployment and general cPanel setup.

## Upload the website now

1. Build the ZIP, if it has not already been built:

   ```bash
   python3 scripts/package_chat_site.py
   ```

   The result is `out/oserai-chat-cpanel.zip`. It contains `index.html`, `api.php`,
   the `assets/` directory, `.htaccess` and `.user.ini` at its root. No build step,
   Node.js application, database or inference credentials are needed on cPanel.

2. In **cPanel → Domains**, find `chat.oserai.network` and note its **Document
   Root**. Use that exact directory in File Manager; it may differ from
   `public_html/chat`. If the subdomain has not been created, give it its own
   document root. [Namecheap subdomain instructions](https://www.namecheap.com/support/knowledgebase/article.aspx/9190/29/how-to-create-a-subdomain-in-cpanel/)

3. Open that document root in **File Manager**, upload the ZIP, and extract it
   there. `index.html` must be directly inside the document root. Show hidden
   files in File Manager to confirm `.htaccess` and `.user.ini` were extracted.
   Remove the upload ZIP after extraction. Keep a copy of any previous website
   files before replacing them.

4. Select **PHP 8.2 or newer** and enable **cURL** in the account's PHP settings.

5. Connect the subdomain's **public DNS** to the hosting account. For a domain
   using Namecheap BasicDNS (`dns1.registrar-servers.com` and
   `dns2.registrar-servers.com`), open **Namecheap → Domain List → Manage
   oserai.network → Advanced DNS → Host Records → Add New Record**:

   | Field | Value |
   | --- | --- |
   | Type | A Record |
   | Host | `chat` |
   | Value | Your hosting account's Shared IP Address, shown in cPanel |
   | TTL | Automatic |

   Save the record. The Host field is `chat`. Creating the website in cPanel
   sets up its document root; when BasicDNS serves the public domain, that
   public A record must also be created in Namecheap's Advanced DNS.
   [Namecheap A-record instructions](https://www.namecheap.com/support/knowledgebase/article.aspx/319/2237/how-can-i-set-up-an-a-address-record-for-my-domain/)

6. After DNS resolves to the hosting account, enable the subdomain's SSL
   certificate and **Force HTTPS Redirect**, then visit
   **https://chat.oserai.network**.

The welcome screen, suggestions, history search, mobile layout, export and
privacy controls work immediately. Sending messages becomes available when
the gateway reports an available model. If `api.php?action=status` downloads
or displays PHP source, PHP is not enabled for that document root; fix that
before adding any private configuration.

### If the URL does not open after uploading

- **DNS error / NXDOMAIN / “server IP address could not be found”:** public DNS
  has no usable address for the subdomain. Add the A record above. Uploading or
  extracting files does not create a BasicDNS host record.
- **Hosting default page or 404:** compare the domain's document root in
  cPanel → Domains with the File Manager extraction directory. `index.html`
  should be directly inside that root.
- **Certificate warning:** first confirm DNS points to this hosting account,
  then complete SSL setup for `chat.oserai.network` in cPanel.
- **500 error:** inspect cPanel's Errors log for the exact cause, including
  any `.htaccess` directive rejected by this hosting account.
- **OserAI loads with “coming online”:** the website is working; the inference
  service connection still needs configuration using the steps below.

To check public DNS from a terminal:

```bash
dig chat.oserai.network A
```

An answer with the hosting IP means DNS is connected. `status: NXDOMAIN` means
the DNS name does not exist at that resolver. If the authoritative nameserver
already has the correct record but a public resolver still returns NXDOMAIN,
its earlier negative lookup can remain cached until that cache expires.

## How the connection works

```text
Browser at chat.oserai.network
  -> api.php on Namecheap cPanel (same origin, PHP session, CSRF protection)
  -> OserAI Chat service on an inference host or VPS (authenticated HTTPS)
  -> Parallax cluster or local Ollama model (model inference)
```

Namecheap Shared Hosting's Python application feature supports WSGI and
explicitly excludes ASGI applications. The streaming FastAPI service therefore
runs on the inference host or a VPS; the public website stays on cPanel.
[Namecheap Python hosting documentation](https://www.namecheap.com/support/knowledgebase/article.aspx/10048/2182/how-to-work-with-python-app/)

The chat connection is independent of batch-agent polling. It does not expose
the coordinator's administrative, training, batch or node-registration APIs to
website visitors. The model runtime and public chat service can run on the same
host, so the model runtime's port can remain private.

## Run the Groq pilot on Render

The pilot uses the existing cPanel site, a Free Render web service for the chat
gateway, and Groq's Qwen 3.8 27B model for replies.

1. Create an API key in the [Groq console](https://console.groq.com/keys).
2. Select a GitHub repository and branch containing this project and its
   `render.yaml` Blueprint.

3. In Render, choose **New → Blueprint**, connect that repository, and
   select the branch. Render reads `render.yaml` from the repository root.
   Enter the Groq key as `OSERAI_CHAT_GROQ_API_KEY` when prompted and deploy the
   Free service. Render generates a separate `OSERAI_CHAT_GATEWAY_KEY`. Copy its
   value from the deployed service's **Environment** tab. Open the service's
   `/health` URL and check for `{"ok":true}`.
4. In cPanel, upload the current `web/chat/public/api.php` to the chat subdomain's
   document root. Preserve any existing `OSERAI_PRIVATE_CONFIG` path override.
   Upload `web/chat/public/assets/app.js` to the subdomain's `assets/` directory
   to display clickable guide links in chat replies.
   In File Manager, turn on **Show Hidden Files**. Create or update
   `.config/oserai/chat.php` under your cPanel home directory, outside the
   document root, and restrict access to the hosting account (normally `600`).

   ```php
   <?php
   return [
       'gateway_url' => 'https://YOUR-RENDER-SERVICE.onrender.com',
       'gateway_key' => 'GENERATED-OSERAI-CHAT-GATEWAY-KEY',
       'allowed_origin' => 'https://chat.oserai.network',
       'allow_local_http' => false,
       'gateway_stream' => false,
   ];
   ```

   The `gateway_url` is the base URL shown in Render. The Groq key belongs in
   Render's environment. The cPanel file holds the generated gateway key.
5. Open [OserAI Chat](https://chat.oserai.network/) in a private window. Send
   “Remember the word cedar,” then ask “What word did I give you?” Groq returns
   each completed reply as a single response. If the site says chat is coming
   online after an idle period, open the Render `/health` URL, wait for it to
   respond, and reload OserAI Chat.

If Render's `/health` returns `{"ok":true}` and chat is still unavailable, check
**cPanel → Metrics → Errors** for an `OserAI Chat` entry. The proxy looks for
`.config/oserai/chat.php` under your cPanel account home even when PHP omits
`HOME`. To supply the path explicitly, copy **Home Directory** from cPanel's
**General Information** panel, append `/.config/oserai/chat.php`, and set
`OSERAI_PRIVATE_CONFIG` near the top of the uploaded `api.php` to that absolute
path.

The Blueprint limits the pilot to two concurrent requests, 512 output tokens
per reply, and 30 requests per visitor per day in anonymous mode. Groq also
enforces shared account limits. Render Free can take about a minute to wake,
and its local SQLite usage counters reset when the service spins down or redeploys.
[Render free tier](https://render.com/docs/free) ·
[Groq rate limits](https://console.groq.com/docs/rate-limits)

### Enable invite-only accounts and token limits

1. Deploy the version with pilot account support to Render. Upload the updated
   `index.html`, `api.php`, `assets/app.js` and `assets/app.css` to the chat
   subdomain's document root in cPanel. The [upload ZIP instructions](#upload-the-website-now)
   include all four files. Extracted public files should have permissions
   `0644`. Preserve any local path override in `api.php` and keep the private
   `chat.php` settings with `gateway_stream` set to `false`.
2. Create a Postgres database. A [Neon Free project](https://neon.com/pricing)
   provides persistent storage for the pilot. In the Neon console, select
   **Connect** and copy the Postgres connection string, including its TLS
   settings. The Render service creates the quota table and index using this
   database role. [Neon connection guide](https://neon.com/docs/connect/connect-from-any-app)
3. Generate a sign-in code for yourself and each invited tester. Use a stable
   account ID per tester. Run this from the repository root:

   ```bash
   python3 -m scripts.create_chat_invites owner tester-1 tester-2
   ```

   Keep the codes private. Give each tester their code directly.
4. In **Render → oserai-chat-pilot → Environment**, add the following
   settings and save and deploy them together. [Render environment settings](https://render.com/docs/configure-environment-variables)

   | Setting | Value |
   | --- | --- |
   | `OSERAI_CHAT_DATABASE_URL` | The Postgres connection string |
   | `OSERAI_CHAT_PILOT_ACCOUNTS` | The JSON line printed by the invite command |
   | `OSERAI_CHAT_ACCOUNTS_REQUIRED` | `1` |

5. Open [OserAI Chat](https://chat.oserai.network/) in a private window. Check
   that sending requires sign-in, then sign in and send a message. Check that
   the remaining allowance drops. Use the same code on a phone to confirm it
   shows the same balance.

Each account gets 5,000 tokens per UTC day. The chat displays the next reset
in the browser's local time. Across all accounts, the gateway admits up to
100,000 tokens per rolling 24 hours and 6,000 per rolling minute, plus at
most 800 requests per 24 hours and 20 per minute. The gateway reserves tokens
before calling Groq and reconciles the reservation using reported input and
output tokens. Input includes any conversation history and OserAI guide
excerpts sent to the model. If provider usage is unknown, the reservation
remains charged for the applicable allowance window. Conversations are
still saved in the browser where they were created.

To add a tester later, generate a code for that account ID and merge its
entry into the existing JSON in Render. To revoke access, remove the account's
entry. To replace a code, generate one for the same ID and replace only its
digest. Redeploy after editing the JSON. Replacing a digest revokes existing
sessions for that ID and preserves its token balance.

## Enable the optional Guide preview

Run Colibri with a compatible GGUF and the model ID `oserai-guide-v0.1`.
Set `COLI_API_KEY` in the Colibri service environment to a separate random key
of at least 32 characters. See the [adaptation guide](model-adaptation.md) for
the local export workflow. The pilot artifacts are for evaluation.

Keep Colibri on loopback and place a stable HTTPS proxy in front of it for a
remote gateway. Forward `POST /v1/chat/completions` to Colibri. Set
`OSERAI_CHAT_GUIDE_URL` and `OSERAI_CHAT_GUIDE_API_KEY` on the chat gateway to
the proxy URL and the runtime key. Type `/guide` followed by a question to use
the preview for one reply. This path uses the same invite accounts and quotas
as regular chat.

## Connect a self-hosted inference service

### On the inference host

1. Start a Parallax cluster using the [MVP guide](parallax-mvp.md). Verify the
   advertised model using `GET /v1/models`, then verify a real completion.
   Set `OSERAI_CHAT_MODEL_NAME` to the exact advertised model ID. The sample
   Qwen 0.6B model is a connectivity test; evaluate a model's answer quality on
   the available hardware before choosing it for end users.

2. Install the small chat service separately from Parallax's GPU environment:

   ```bash
   python3.12 -m venv .venv-chat
   source .venv-chat/bin/activate
   pip install -r requirements-chat.txt
   cp chat_service/.env.example chat_service/.env
   ```

3. Generate a private gateway key locally:

   ```bash
   python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
   ```

   Put it in `OSERAI_CHAT_GATEWAY_KEY` in the ignored `.env` file. Configure
   `OSERAI_CHAT_PARALLAX_URL`, `OSERAI_CHAT_MODEL_NAME` and `OSERAI_CHAT_MODEL_LABEL`.
   If Parallax itself is behind an authenticated proxy, also set the optional
   `OSERAI_CHAT_PARALLAX_API_KEY`. Never put these keys in JavaScript or a ZIP.

4. Run a **single** gateway process:

   ```bash
   uvicorn chat_service.main:app --env-file chat_service/.env \
     --host 127.0.0.1 --port 8010 --workers 1
   ```

   For an existing Linux Docker host, `deploy/Dockerfile.chat` and
   `deploy/docker-compose.chat.yaml` are an alternative. Copy the environment
   file to ignored `deploy/chat.env`, then run:

   ```bash
   docker compose -f deploy/docker-compose.chat.yaml up -d --build
   ```

   This configuration uses Linux host networking to reach a local Parallax
   instance and keeps the gateway on loopback. It is independent of the
   existing coordinator deployment.

5. Give the gateway an HTTPS address reachable from Namecheap, such as a
   separate subdomain pointing at the inference/VPS host. Keep
   `chat.oserai.network` pointed at Namecheap. The provided
   `deploy/Caddyfile.chat.example` shows the reverse proxy and request-body
   limit. Add that site block to the host's existing Caddy configuration after
   substituting the real gateway domain. Caddy flushes responses with
   `Content-Type: text/event-stream` immediately.
   [Caddy streaming behavior](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming)

### In cPanel

6. Using File Manager with hidden files visible, create
   `/home/YOUR_CPANEL_USERNAME/.config/oserai/chat.php` **outside all website
   document roots**. Use the content from `web/chat/config.example.php`:

   ```php
   <?php
   return [
       'gateway_url' => 'https://YOUR-CHAT-GATEWAY-DOMAIN',
       'gateway_key' => 'THE-SAME-PRIVATE-KEY-AS-OSERAI_CHAT_GATEWAY_KEY',
       'allowed_origin' => 'https://chat.oserai.network',
       'allow_local_http' => false,
   ];
   ```

   The URL is the gateway's base URL, without `/api/chat`. Restrict this file
   to the hosting account, normally permissions `600`. The PHP process must
   be able to read it. The application uses `$HOME/.config/oserai/chat.php` by
   default. If PHP's `HOME` environment variable is unavailable, edit
   `OSERAI_PRIVATE_CONFIG` at the top of `api.php` to that absolute private
   path, or set `OSERAI_CHAT_CONFIG` in the server environment. The pointer
   contains a path, never a key. Keep that path edit when replacing `api.php`
   in later uploads.

7. Reload the website. It should change to **“Ready to explore.”** Send a
   message, ask a follow-up, then test Stop and Try again. Test on a phone too.

## What this release includes

- Responsive chat interface, message suggestions, safe text/Markdown rendering,
  streamed replies, copy, export, stop and retry.
- Conversations and search stored in the visitor's browser. No account login,
  cross-device sync or payment collection is included in this initial release.
- An HTTP-only session cookie and CSRF checks on cPanel. The browser never
  receives the private gateway key or Parallax address.
- Private bearer authentication between cPanel and the chat service. The
  Parallax URL and served model are configured by the server, never by visitors.
- Defaults of four concurrent replies, ten requests/minute per anonymous
  network identity, and one hundred requests/day per session. Environment
  settings in `ChatConfig` control the limits. Requests that reached the gateway
  count toward quotas even if generation is interrupted.
- A separate SQLite usage database with request status, duration and token
  counts when reported by Parallax. Missing counts stay unknown. The gateway
  stores no message content in this database. Usage belongs to the cluster;
  there is no automatic split into contributor payouts.

Daily limits are anonymous session limits, not authenticated account quotas.
Conversation histories are visible to anyone using the same browser profile.
Clearing site data removes those histories and changes the session identity.

## Verification and hosting limits

Automated gateway tests use a simulated Parallax HTTP service. Browser tests
exercise desktop/mobile layouts, streaming, conversation context, local history,
search, stop/retry, copy/export, offline states and deletion. PHP 8.3 checks
exercise the real proxy code against a simulated HTTP service, including
private credentials, cookies, CSRF/origin validation, UTF-8 streaming and 429s.

Namecheap/LiteSpeed can add buffering or timeouts after PHP. The package turns
off PHP output buffering and compression, but incremental delivery must still
be verified on **this hosting account** after upload. If text arrives only at
the end, the UI still reads the reply, but it won't feel streamed. Ask hosting
support about SSE/buffering for this subdomain before increasing traffic.
Namecheap documents account entry-process limits, so each long-running PHP
request should be considered when sizing this small beta.
[Namecheap resource limits](https://www.namecheap.com/support/knowledgebase/article.aspx/1127/103/a-handy-guide-to-resource-limits-or-what-is-lve/)

The concurrency gate is per process; keep `--workers 1` until there is a shared
capacity controller. Cancellation closes the upstream stream, but how quickly
GPU work stops depends on Parallax and the hosting proxy detecting disconnects.

Run the Python suite with `python -m pytest`. For optional local browser/PHP
checks with an existing Chrome installation:

```bash
npm --prefix /tmp/oserai-chat-checks install playwright @php-wasm/cli
NODE_PATH=/tmp/oserai-chat-checks/node_modules node web/chat/tests/browser.cjs
NODE_PATH=/tmp/oserai-chat-checks/node_modules node web/chat/tests/php.cjs
```

The test-only simulated services are excluded from the upload ZIP. No live
Parallax, Namecheap deployment, SSL issuance or DNS changes are performed by
these checks. Once the files and real gateway are connected, verify the public
URL and live cluster before inviting end users.
