/* PHP 8.3 compatibility and proxy checks using the optional @php-wasm/node
   interpreter. No inference credentials or production services are used. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");

async function run() {
  const {PHP, FileLockManagerInMemory} = await import(require.resolve("@php-wasm/universal"));
  const {loadNodeRuntime, withNetworking} = await import(require.resolve("@php-wasm/node"));
  let upstreamMode = "ready", calls = 0, accountsRequired = false;
  const key = "local-php-gateway-test-key-never-used-in-production";
  const accessCode = "A".repeat(43);
  const proof = "alice." + "a".repeat(64);
  const quota = {limit: 5000, remaining: 4900, resets_at: "2026-11-20T00:00:00Z"};
  const upstream = http.createServer(async (req, res) => {
    calls++;
    assert.equal(req.headers.authorization, `Bearer ${key}`);
    const signedIn = req.headers["x-chat-account"] === proof;
    if (req.url === "/api/status") {
      res.writeHead(200, {"Content-Type": "application/json"});
      return res.end(JSON.stringify({ready: true, model_label: "OserAI Chat", max_input_chars: 16000,
        accounts_required: accountsRequired, signed_in: signedIn, quota: signedIn ? quota : null}));
    }
    assert.match(req.headers["x-visitor-id"], /^[a-f0-9]{64}$/);
    assert.match(req.headers["x-visitor-network"], /^[a-f0-9]{64}$/);
    let data = ""; for await (const chunk of req) data += chunk;
    const payload = JSON.parse(data);
    if (req.url === "/api/login") {
      res.writeHead(payload.code === accessCode ? 200 : 401, {"Content-Type": "application/json"});
      return res.end(JSON.stringify(payload.code === accessCode ? {account_credential: proof} : {}));
    }
    assert.deepEqual(payload.messages, [{role: "user", content: "hello"}]);
    if (accountsRequired && !signedIn) {
      res.writeHead(401, {"Content-Type": "application/json"});
      return res.end(JSON.stringify({message: "Sign in to chat."}));
    }
    if (upstreamMode === "busy") { res.writeHead(429, {"Content-Type": "application/json"}); return res.end(JSON.stringify({message: "Please try again later."})); }
    if (upstreamMode.startsWith("buffered")) {
      assert.equal(payload.stream, false);
      assert.equal(req.headers.accept, "application/json");
      res.writeHead(200, {"Content-Type": "application/json"});
      return res.end(JSON.stringify(upstreamMode === "buffered-bad" ? {unexpected: "PRIVATE ENGINE DETAILS"} : {text: "A complete reply 🌱", completion_tokens: 8, elapsed_s: 1.2, request_id: "request-1", quota: signedIn ? quota : null}));
    }
    assert.equal(payload.stream, true);
    res.writeHead(200, {"Content-Type": "text/event-stream"});
    res.write('event: token\ndata: {"text":"A streamed reply 🌱"}\n\n');
    setTimeout(() => res.end('event: done\ndata: {"completion_tokens":5}\n\n'), 35);
  });
  await new Promise((resolve) => upstream.listen(0, "127.0.0.1", resolve));
  const php = new PHP(await loadNodeRuntime("8.3", {fileLockManager: new FileLockManagerInMemory(), emscriptenOptions: {...await withNetworking(), processId: 1}}));
  php.mkdirTree("/www"); php.mkdirTree("/private");
  php.writeFile("/www/api.php", fs.readFileSync(path.resolve(__dirname, "../public/api.php")));
  const config = {gateway_url: `http://127.0.0.1:${upstream.address().port}`, gateway_key: key, allowed_origin: "http://chat.test", allow_local_http: true};
  php.writeFile("/private/config.php", `<?php return ${phpArray(config)};`);
  const base = {scriptPath: "/www/api.php", protocol: "http", env: {OSERAI_CHAT_CONFIG: "/private/config.php"}, $_SERVER: {REMOTE_ADDR: "127.0.0.1"}};
  const status = await php.run({...base, relativeUri: "/api.php?action=status", method: "GET"});
  assert.equal(status.httpStatusCode, 200, status.text);
  const statusBody = JSON.parse(status.text);
  assert.equal(statusBody.ready, true);
  assert.equal(status.text.includes(key), false);
  const cookie = status.headers["set-cookie"][0].split(";")[0];
  const headers = {"Content-Type": "application/json", "Origin": "http://chat.test", "Cookie": cookie, "X-CSRF-Token": statusBody.csrf_token};
  const post = { ...base, relativeUri: "/api.php?action=chat", method: "POST", body: new TextEncoder().encode(JSON.stringify({messages: [{role: "user", content: "hello"}]})), headers };
  const rejected = await php.run({...post, headers: {...headers, "X-CSRF-Token": "wrong"}});
  assert.equal(rejected.httpStatusCode, 403);
  const wrongOrigin = await php.run({...post, headers: {...headers, Origin: "https://another-site.example"}});
  assert.equal(wrongOrigin.httpStatusCode, 403);
  assert.equal(calls, 1, "invalid browser requests never reach the upstream");
  const result = await php.run(post);
  assert.equal(result.httpStatusCode, 200, result.text);
  assert.match(result.text, /event: token/);
  assert.match(result.text, /A streamed reply 🌱/);
  assert.match(result.text, /event: done/);
  config.gateway_stream = false;
  php.writeFile("/private/config.php", `<?php return ${phpArray(config)};`);
  upstreamMode = "buffered";
  const buffered = await php.run(post);
  assert.equal(buffered.httpStatusCode, 200, buffered.text);
  assert.match(buffered.text, /event: start/);
  assert.match(buffered.text, /A complete reply 🌱/);
  assert.match(buffered.text, /event: done/);
  assert.match(buffered.text, /"completion_tokens":8/);
  upstreamMode = "buffered-bad";
  const malformed = await php.run(post);
  assert.equal(malformed.httpStatusCode, 503, malformed.text);
  assert.equal(malformed.text.includes("PRIVATE ENGINE DETAILS"), false);
  assert.equal(malformed.text.includes("event: done"), false);
  upstreamMode = "busy";
  const busy = await php.run(post);
  assert.equal(busy.httpStatusCode, 429, busy.text);
  assert.equal(JSON.parse(busy.text).message, "Please try again later.");
  upstreamMode = "buffered";
  accountsRequired = true;
  const guest = await php.run({...post, headers: {...headers, "X-Chat-Account": proof}});
  assert.equal(guest.httpStatusCode, 401, guest.text);
  const signIn = { ...post, relativeUri: "/api.php?action=login", body: new TextEncoder().encode(JSON.stringify({code: accessCode})) };
  const badLogin = await php.run({...signIn, headers: {...headers, "X-CSRF-Token": "wrong"}});
  assert.equal(badLogin.httpStatusCode, 403);
  const signedIn = await php.run(signIn);
  assert.equal(signedIn.httpStatusCode, 200, signedIn.text);
  assert.equal(signedIn.text.includes(proof), false);
  assert.equal(signedIn.text.includes(accessCode), false);
  const loginCookie = signedIn.headers["set-cookie"].at(-1).split(";")[0];
  assert.notEqual(loginCookie, cookie);
  const authHeaders = {...headers, "Cookie": loginCookie, "X-CSRF-Token": JSON.parse(signedIn.text).csrf_token};
  assert.notEqual(authHeaders["X-CSRF-Token"], headers["X-CSRF-Token"]);
  const accountStatus = await php.run({...base, relativeUri: "/api.php?action=status", method: "GET", headers: {"Cookie": loginCookie}});
  assert.deepEqual(JSON.parse(accountStatus.text).quota, quota);
  const accountReply = await php.run({...post, headers: authHeaders});
  assert.equal(accountReply.httpStatusCode, 200, accountReply.text);
  assert.match(accountReply.text, /"remaining":4900/);
  const logout = await php.run({...post, relativeUri: "/api.php?action=logout", headers: authHeaders});
  assert.equal(logout.httpStatusCode, 200, logout.text);
  const logoutCookie = logout.headers["set-cookie"].at(-1).split(";")[0];
  const signedOut = await php.run({...base, relativeUri: "/api.php?action=status", method: "GET", headers: {"Cookie": logoutCookie}});
  assert.equal(JSON.parse(signedOut.text).signed_in, false);
  const offline = await php.run({...base, relativeUri: "/api.php?action=status", env: {OSERAI_CHAT_CONFIG: "/private/missing.php"}});
  assert.equal(JSON.parse(offline.text).ready, false);
  const home = "/home2/oserai_test";
  const docroot = home + "/public_html/chat";
  php.mkdirTree(docroot);
  php.mkdirTree(home + "/.config/oserai");
  php.writeFile(docroot + "/api.php", fs.readFileSync(path.resolve(__dirname, "../public/api.php")));
  php.writeFile(home + "/.config/oserai/chat.php", `<?php return ${phpArray(config)};`);
  const cpanel = {...base, scriptPath: docroot + "/api.php", relativeUri: "/api.php?action=status", method: "GET"};
  const withoutHome = await php.run({...cpanel, env: {HOME: ""}});
  assert.equal(JSON.parse(withoutHome.text).ready, true, withoutHome.text);
  assert.equal(withoutHome.text.includes(key), false);
  const withOverride = await php.run({...cpanel, env: {HOME: "", OSERAI_CHAT_CONFIG: "/private/missing.php"}});
  assert.equal(JSON.parse(withOverride.text).ready, false);
  console.log("PHP 8.3 checks passed, including account sign-in, session rotation, sign-out, and allowance forwarding.");
  upstream.closeAllConnections(); upstream.close();
}
function phpArray(object) {
  return "[" + Object.entries(object).map(([key, value]) => `'${key}' => ${typeof value === "boolean" ? String(value) : `'${String(value).replaceAll("'", "\\'")}'`}`).join(", ") + "]";
}
run().then(() => process.exit(0), (error) => { console.error(error); process.exit(1); });
