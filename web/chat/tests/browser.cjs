/* Run with Playwright installed locally or in NODE_PATH. All inference here
   is an explicit test fixture, never a mode shipped in the public site. */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");

const root = path.resolve(__dirname, "../public");
const output = process.env.OSERAI_BROWSER_OUTPUT || path.resolve(__dirname, "../../../out/chat-preview");
const csp = fs.readFileSync(path.join(root, ".htaccess"), "utf8")
  .match(/Content-Security-Policy "([^"]+)"/)[1];
fs.mkdirSync(output, {recursive: true});
const previewPrompt = "How do I test two computers running a model together?";
let mode = "ready", requests = [], openStreams = 0;
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/api.php" && url.searchParams.get("action") === "status") {
    res.writeHead(200, {"Content-Type": "application/json"});
    return res.end(JSON.stringify({ready: mode !== "offline", csrf_token: "test-csrf-token", model_label: "OserAI Chat", max_input_chars: 16000}));
  }
  if (url.pathname === "/api.php" && req.method === "POST") {
    let raw = ""; for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw); requests.push(body);
    assert.equal(req.headers["x-csrf-token"], "test-csrf-token");
    if (mode === "busy") { res.writeHead(503, {"Content-Type": "application/json"}); return res.end(JSON.stringify({message: "All chat slots are busy. Please try again shortly."})); }
    res.writeHead(200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"});
    const event = (name, payload) => res.write(`event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`);
    event("start", {model_label: "OserAI Chat"});
    openStreams++;
    const slow = body.messages.at(-1).content === "Slow response";
    const normalParts = [
      "## A starting point\n\n", "**Curiosity** leads to useful ideas. 🌱\n\n",
      "<img src=x onerror=alert(1)>\n\n", "- Ask a clear question.\n- Try one small step.\n\n",
      "```js\nconst idea = 'start small';\n```",
      "\n\n- [OserAI Parallax guide](https://oserai-chat-pilot.onrender.com/guides/parallax-cluster)\n- [Unsafe guide](javascript:alert(1))",
    ];
    const previewParts = [
      "## Start with a small model\n\n",
      "Use a small model and connect both computers to the same local network. This gives you a way to check discovery, model loading, and inference before inviting remote workers.\n\n",
      "1. Start the Parallax scheduler on your first computer.\n2. Join from a separate terminal on each computer.\n3. Wait for both workers to become ready, then try a simple prompt.\n\n",
      "```bash\nparallax run -m Qwen/Qwen3-0.6B -n 2\nparallax join\n```\n\n",
      "If the scheduler stays on one worker, check the local network permissions on each device. Once both workers are visible, generate a short reply to confirm the model is ready.\n\n",
      "- [OserAI Parallax guide](https://oserai-chat-pilot.onrender.com/guides/parallax-cluster)",
    ];
    const parts = slow ? Array(18).fill("One thoughtful idea. ")
      : body.messages.at(-1).content === previewPrompt ? previewParts : normalParts;
    let index = 0;
    const timer = setInterval(() => {
      if (res.destroyed) { clearInterval(timer); return; }
      if (index < parts.length) event("token", {text: parts[index++]});
      else { event("done", {completion_tokens: 32}); res.end(); clearInterval(timer); }
    }, slow ? 700 : 35);
    res.on("close", () => { clearInterval(timer); openStreams--; });
    return;
  }
  const requested = path.resolve(root, "." + url.pathname);
  const file = requested === root ? path.join(root, "index.html") : requested;
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile() || file.endsWith(".php")) { res.writeHead(404); return res.end(); }
  const types = {".html": "text/html", ".css": "text/css", ".js": "application/javascript", ".svg": "image/svg+xml"};
  res.writeHead(200, {
    "Content-Type": types[path.extname(file)] || "text/plain",
    "Content-Security-Policy": csp,
  });
  fs.createReadStream(file).pipe(res);
});

async function checkLayout(page, width) {
  assert.equal(await page.evaluate(() => {
    const main = document.querySelector("main").getBoundingClientRect();
    const scroll = document.querySelector("#conversation-area");
    return main.right <= innerWidth + 1 && main.bottom <= innerHeight + 1
      && scroll.scrollWidth <= scroll.clientWidth + 1;
  }), true, `Layout at ${width}px`);
  const smallTargets = await page.locator("button:visible, a:visible, input:visible, textarea:visible").evaluateAll(
    (controls) => controls.filter((control) => !control.classList.contains("skip-link")
      && control.getBoundingClientRect().height < 43.5).map((control) => control.id || control.textContent.trim()),
  );
  assert.deepEqual(smallTargets, [], `Touch targets at ${width}px`);
}

async function checkRedesign(browser, base, width, height) {
  const context = await browser.newContext({
    viewport: {width, height}, isMobile: width < 761, hasTouch: width < 761, reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const screenshot = (name) => page.screenshot({path: path.join(output, `${name}-${width}.png`)});
  try {
    await page.goto(base);
    await page.waitForFunction(() => document.querySelector("#network-status").classList.contains("ready"));
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator("#welcome-title").isVisible(), true);
    assert.equal(await page.locator(".suggestion").count(), 4);
    assert.equal(await page.locator("#chat-list strong").textContent(), "Nothing saved yet");
    assert.equal(await page.locator(".welcome-mark").evaluate((mark) => getComputedStyle(mark).animationName), "none");
    await checkLayout(page, width);
    if (width < 761) {
      assert.equal(await page.locator(".composer-dock").evaluate((dock) => (
        Math.abs(dock.getBoundingClientRect().bottom - innerHeight) < 2
      )), true, "The mobile welcome composer stays at the bottom");
    }
    await screenshot("welcome");

    await page.keyboard.press("Tab");
    assert.equal(await page.locator(".skip-link").evaluate((anchor) => document.activeElement === anchor), true);
    await page.keyboard.press("Enter");
    const prompt = page.locator("#prompt");
    assert.equal(await prompt.evaluate((input) => document.activeElement === input), true);
    assert.equal(await prompt.getAttribute("rows"), width < 761 ? "1" : "3");

    if (width >= 761) {
      await page.getByRole("link", {name: "How replies are made", exact: true}).click();
      assert.equal(await page.locator("#how-replies-are-made").evaluate((section) => (
        section.getBoundingClientRect().top < innerHeight
      )), true);
      if (width === 1440) await screenshot("info");
    } else {
      assert.equal(await prompt.evaluate((input) => parseFloat(getComputedStyle(input).fontSize) >= 16), true);
      await page.getByRole("button", {name: "Open conversations"}).click();
      assert.equal(await page.locator("#sidebar").evaluate((sidebar) => !sidebar.inert), true);
      await screenshot("drawer");
      await page.keyboard.press("Escape");
      await page.waitForFunction(() => document.activeElement.id === "menu-button");
    }

    await prompt.fill(previewPrompt);
    await prompt.press("Shift+Enter");
    assert.equal(await prompt.inputValue(), previewPrompt + "\n");
    await prompt.press("Enter");
    await page.getByRole("button", {name: "Copy reply"}).waitFor();
    assert.equal(await page.locator("#conversation-title").textContent(), previewPrompt);
    assert.equal(await page.locator(".message-assistant ol li").count(), 3);
    assert.match(await page.locator(".message-assistant pre").textContent(), /parallax join/);
    assert.equal(await page.locator(".chat-item-main").getAttribute("aria-current"), "page");
    assert.equal(await prompt.getAttribute("rows"), "1");
    await checkLayout(page, width);
    await page.locator("#conversation-area").evaluate((area) => { area.scrollTop = 0; });
    await screenshot("conversation");
    assert.equal(await page.locator(".composer-dock").evaluate((dock) => (
      Math.abs(dock.getBoundingClientRect().bottom - innerHeight) < 2
    )), true, "The conversation composer stays at the bottom while scrolling");

    await page.reload();
    const saved = page.getByRole("button", {name: previewPrompt, exact: true});
    if (width < 761) await page.getByRole("button", {name: "Open conversations"}).click();
    await saved.click();
    assert.equal(await page.locator(".message").count(), 2);
    await page.getByRole("button", {name: "Delete current conversation"}).click();
    const dialog = page.getByRole("dialog", {name: "Clear this conversation?"});
    await dialog.waitFor();
    assert.equal(await dialog.getAttribute("aria-modal"), "true");
    assert.equal(await dialog.locator(".danger-eyebrow").textContent(), "This can’t be undone");
    assert.equal(await dialog.locator("#confirm-description").textContent(), "This removes the saved conversation from this browser.");
    await page.waitForFunction(() => document.activeElement.id === "confirm-cancel");
    await page.keyboard.press("Shift+Tab");
    assert.equal(await page.evaluate(() => document.activeElement.id), "confirm-delete");
    await page.keyboard.press("Tab");
    assert.equal(await page.evaluate(() => document.activeElement.id), "confirm-cancel");
    await screenshot("clear-dialog");
    await page.keyboard.press("Escape");
    await dialog.waitFor({state: "hidden"});
    await page.waitForFunction(() => document.activeElement.id === "delete-current-chat");

    await page.keyboard.press(width === 1024 ? "Control+Shift+O" : "Meta+Shift+O");
    assert.equal(await page.locator("#welcome-title").isVisible(), true);
    assert.equal(await page.locator(".chat-item").count(), 1);
    await prompt.fill("Slow response");
    await prompt.press("Enter");
    await page.getByText("Preparing your reply…").waitFor();
    await screenshot("waiting");
    await page.getByRole("button", {name: "Stop generating"}).click();
    await page.getByText("Reply stopped.", {exact: true}).first().waitFor();

    await page.getByRole("button", {name: "Delete current conversation"}).click();
    await page.getByRole("button", {name: "Delete conversation", exact: true}).click();
    assert.equal(await page.locator("#welcome-title").isVisible(), true);
    assert.equal(await page.locator(".chat-item").count(), 1);
  } finally {
    await context.close();
  }
}

async function run() {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({channel: "chrome", headless: true});
  const errors = [];
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, permissions: ["clipboard-read", "clipboard-write"]});
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error" && /Content Security Policy|violates.*directive/i.test(message.text())) errors.push(message.text()); });
  try {
    let allowStatus;
    const statusReleased = new Promise((resolve) => { allowStatus = resolve; });
    await page.route("**/api.php?action=status", async (route) => {
      await statusReleased;
      await route.continue();
    }, {times: 1});
    await page.goto(base, {waitUntil: "domcontentloaded"});
    assert.match(await page.locator("#network-status").textContent(), /Connecting/);
    allowStatus();
    await page.waitForFunction(() => document.querySelector("#network-status").classList.contains("ready"));
    assert.equal(await page.locator("#welcome-title").textContent(), "What’s on your mind today?");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(output, "desktop.png"), fullPage: true});

    await page.getByRole("button", {name: /Make something clear/}).click();
    assert.match(await page.locator("#prompt").inputValue(), /neural networks/);
    await page.locator("#prompt").fill("Help me explore an idea.");
    await page.getByRole("button", {name: "Send message", exact: true}).click();
    await page.getByRole("button", {name: "Copy reply"}).waitFor();
    assert.equal(await page.locator(".message-assistant img").count(), 0);
    assert.match(await page.locator(".message-assistant").innerText(), /<img src=x onerror=alert\(1\)>/);
    assert.equal(await page.locator(".message-assistant strong").innerText(), "Curiosity");
    const guide = page.getByRole("link", {name: "OserAI Parallax guide"});
    assert.equal(await guide.getAttribute("href"), "https://oserai-chat-pilot.onrender.com/guides/parallax-cluster");
    assert.equal(await page.getByRole("link", {name: "Unsafe guide"}).count(), 0);
    await page.getByRole("button", {name: "Copy reply"}).click();
    await page.getByText("Copied", {exact: true}).waitFor();
    await page.screenshot({path: path.join(output, "conversation.png"), fullPage: true});

    await page.locator("#prompt").fill("Can you expand on that?");
    await page.locator("#prompt").press("Enter");
    await page.waitForFunction(() => document.querySelectorAll(".message-assistant .message-actions").length === 2);
    assert.equal(requests.at(-1).messages.length, 3);
    assert.deepEqual(requests.at(-1).messages.map((m) => m.role), ["user", "assistant", "user"]);
    const downloadWait = page.waitForEvent("download");
    await page.getByRole("button", {name: "Download conversation"}).click();
    const download = await downloadWait;
    assert.match(download.suggestedFilename(), /^oserai-/);
    await page.reload();
    assert.equal(await page.locator("#chat-list .chat-item").count(), 1);
    await page.locator(".chat-item-main").click();
    assert.equal(await page.locator(".message").count(), 4);
    await page.locator("#chat-search").fill("nothing-matches-this");
    assert.equal(await page.locator(".history-empty").innerText(), "No conversations found.");
    await page.locator("#chat-search").fill("");

    await page.locator("#new-chat").click();
    await page.locator("#prompt").fill("Slow response");
    await page.getByRole("button", {name: "Send message", exact: true}).click();
    await page.waitForFunction(() => document.querySelector(".message-assistant .message-body").textContent.includes("thoughtful"));
    await page.getByRole("button", {name: "Stop generating"}).click();
    await page.getByText("Reply stopped.", {exact: true}).first().waitFor();
    await page.waitForTimeout(150);
    assert.equal(openStreams, 0, "cancelling closes the HTTP response");
    await page.locator("#prompt").fill("A different question");
    await page.getByRole("button", {name: "Send message", exact: true}).click();
    await page.getByRole("button", {name: "Copy reply"}).last().waitFor();
    assert.equal(requests.at(-1).messages.length, 1, "incomplete replies are excluded from subsequent context");

    mode = "busy";
    await page.locator("#new-chat").click();
    await page.locator("#prompt").fill("Try a busy cluster");
    await page.getByRole("button", {name: "Send message", exact: true}).click();
    await page.locator(".message-error").waitFor();
    assert.match(await page.locator(".message-error").textContent(), /busy/);
    mode = "ready";
    await page.getByRole("button", {name: "Try again", exact: true}).click();
    await page.getByRole("button", {name: "Copy reply"}).waitFor();
    assert.equal(await page.locator(".message-user").count(), 1);

    const mobile = await browser.newContext({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});
    const phone = await mobile.newPage(); phone.on("pageerror", (error) => errors.push(error.message));
    await phone.goto(base);
    await phone.waitForFunction(() => document.querySelector("#network-status").classList.contains("ready"));
    assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.equal(await phone.locator("#sidebar").evaluate((node) => node.inert), true);
    await phone.screenshot({path: path.join(output, "mobile.png"), fullPage: true});
    await phone.getByRole("button", {name: "Open conversations"}).click();
    assert.equal(await phone.locator("#sidebar").evaluate((node) => node.inert), false);
    await phone.keyboard.press("Escape");
    assert.equal(await phone.locator("#sidebar").evaluate((node) => node.inert), true);
    await phone.locator("#prompt").fill("A mobile question");
    await phone.getByRole("button", {name: "Send message", exact: true}).click();
    await phone.getByRole("button", {name: "Copy reply"}).waitFor();
    assert.equal(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);

    mode = "offline";
    const offlineContext = await browser.newContext({viewport: {width: 1440, height: 1000}});
    const offline = await offlineContext.newPage();
    await offline.goto(base);
    await offline.getByText("Chat is coming online. Please check back soon.").waitFor();
    assert.match(await offline.locator("#network-status").textContent(), /Offline/);
    await offline.locator("#prompt").fill("I have a question");
    assert.equal(await offline.locator("#send-button").isDisabled(), true);
    await offline.screenshot({path: path.join(output, "coming-online.png"), fullPage: true});

    mode = "ready";
    await page.locator("#about-button").click();
    await page.getByRole("button", {name: "Clear all saved conversations"}).click();
    assert.equal(await page.locator("#confirm-description").textContent(), "This removes every saved conversation from this browser.");
    await page.locator("#confirm-delete").click();
    assert.equal(await page.locator(".chat-item").count(), 0);
    assert.equal(await page.locator("#welcome").isVisible(), true);
    for (const [width, height] of [[1440, 1000], [1024, 900], [390, 844]]) {
      await checkRedesign(browser, base, width, height);
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "browser-checks.json"), JSON.stringify({
      passed: true, widths: [1440, 1024, 390],
      checks: ["safe Markdown", "streaming", "context", "copy/export", "history/search", "stop/retry",
        "offline", "clear history", "keyboard shortcuts", "skip link", "sticky composer", "dialog focus", "tap targets"],
      consoleErrors: errors,
    }, null, 2));
    console.log("Browser checks passed at 1440px, 1024px, and 390px.");
  } finally {
    await browser.close(); server.closeAllConnections(); await new Promise((resolve) => server.close(resolve));
  }
}
run().catch((error) => { console.error(error); server.closeAllConnections(); server.close(); process.exitCode = 1; });
