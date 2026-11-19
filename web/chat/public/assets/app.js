"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE = "oserai.chat.v1";
  let conversations = [], currentId = null, generation = null, csrf = "", ready = false, maxInput = 16000;
  let toastTimer, deleteAction;
  const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="assets/icons.svg#${name}"></use></svg>`;
  const escape = (text) => String(text).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const uid = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  function toast(message) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
  }

  function persist() {
    try { localStorage.setItem(STORAGE, JSON.stringify(conversations.slice(0, 30))); }
    catch { toast("This browser couldn't save the conversation."); }
  }

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE) || "[]");
      if (!Array.isArray(saved)) return;
      conversations = saved.filter((chat) => chat && typeof chat.id === "string" && typeof chat.title === "string" && Array.isArray(chat.messages)).slice(0, 30).map((chat) => ({
        id: chat.id, title: chat.title.slice(0, 100), updatedAt: Number(chat.updatedAt) || Date.now(),
        messages: chat.messages.filter((m) => m && ["user", "assistant"].includes(m.role) && typeof m.content === "string").slice(-128).map((m) => ({
          id: typeof m.id === "string" ? m.id : uid(), role: m.role, content: m.content.slice(0, 100000),
          state: m.state === "complete" ? "complete" : "interrupted", include: m.state === "complete" && m.include !== false,
        })),
      }));
      for (const chat of conversations) for (let i = 1; i < chat.messages.length; i++) {
        if (chat.messages[i].role === "assistant" && chat.messages[i].state !== "complete" && chat.messages[i - 1].role === "user") chat.messages[i - 1].include = false;
      }
    } catch { conversations = []; }
  }

  function current() { return conversations.find((chat) => chat.id === currentId); }

  function sidebar(open) {
    $("sidebar").classList.toggle("open", open);
    $("sidebar-scrim").hidden = !open;
    $("menu-button").setAttribute("aria-expanded", String(open));
    $("sidebar").inert = innerWidth <= 760 && !open;
    document.querySelector("main").inert = innerWidth <= 760 && open;
    if (open) $("chat-search").focus();
  }

  function history() {
    const filter = $("chat-search").value.toLowerCase().trim();
    const matches = conversations.filter((chat) => !filter || chat.title.toLowerCase().includes(filter) || chat.messages.some((m) => m.content.toLowerCase().includes(filter)));
    $("chat-list").replaceChildren();
    if (!matches.length) {
      const p = document.createElement("p");
      p.className = "history-empty";
      p.textContent = filter ? "No conversations found." : "A good question is a beginning.\nYour conversations will appear here.";
      $("chat-list").append(p);
    }
    for (const chat of matches) {
      const item = document.createElement("div");
      item.className = `chat-item${chat.id === currentId ? " active" : ""}`;
      const select = document.createElement("button");
      select.className = "chat-item-main";
      select.textContent = chat.title;
      select.title = chat.title;
      if (chat.id === currentId) select.setAttribute("aria-current", "true");
      select.addEventListener("click", () => { stop(); currentId = chat.id; render(); sidebar(false); });
      const remove = document.createElement("button");
      remove.className = "chat-delete";
      remove.innerHTML = icon("trash");
      remove.setAttribute("aria-label", `Delete ${chat.title}`);
      remove.addEventListener("click", () => confirmDelete(false, () => {
        if (currentId === chat.id) { stop(); currentId = null; }
        conversations = conversations.filter((c) => c.id !== chat.id);
        persist(); render();
      }));
      item.append(select, remove); $("chat-list").append(item);
    }
  }

  // Escape all model output first. Only a small, safe Markdown subset is
  // rendered; model HTML and executable links are never inserted as HTML.
  function markdown(text) {
    const guideLink = /^\[([^\]\n]+)\]\(((?:https:\/\/[a-z0-9.-]+|http:\/\/(?:localhost|127\.0\.0\.1))(?::\d{1,5})?\/guides\/[a-z0-9-]+)\)$/i;
    const inline = (value) => {
      const citation = guideLink.exec(value.trim());
      if (citation) {
        return '<a href="' + escape(citation[2]) + '" target="_blank" rel="noopener noreferrer">' + escape(citation[1]) + '</a>';
      }
      return escape(value).replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    };
    return text.split("```").map((part, index) => {
      if (index % 2) return `<pre><code>${escape(part.replace(/^[\w+-]*\n/, ""))}</code></pre>`;
      let html = "", paragraph = [], list = null;
      const flush = () => { if (paragraph.length) { html += `<p>${paragraph.join("<br>")}</p>`; paragraph = []; } };
      const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
      for (const line of part.split("\n")) {
        const heading = line.match(/^#{1,4}\s+(.+)/);
        const bullet = line.match(/^\s*[-*]\s+(.+)/);
        const ordered = line.match(/^\s*\d+\.\s+(.+)/);
        if (heading) { flush(); closeList(); html += `<h3>${inline(heading[1])}</h3>`; }
        else if (bullet || ordered) {
          flush(); const tag = bullet ? "ul" : "ol";
          if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; }
          html += `<li>${inline((bullet || ordered)[1])}</li>`;
        } else if (!line.trim()) { flush(); closeList(); }
        else { closeList(); paragraph.push(inline(line)); }
      }
      flush(); closeList(); return html;
    }).join("");
  }

  async function copy(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      button.innerHTML = `${icon("check")}<span>Copied</span>`;
      setTimeout(() => { if (button.isConnected) button.innerHTML = `${icon("copy")}<span>Copy</span>`; }, 1600);
    } catch { toast("Copy isn't available in this browser. You can select the text instead."); }
  }

  function renderMessages() {
    const chat = current();
    const hasMessages = Boolean(chat?.messages.length);
    $("welcome").hidden = hasMessages;
    $("messages").hidden = !hasMessages;
    $("messages").replaceChildren();
    $("export-chat").disabled = !hasMessages;
    if (!hasMessages) return;
    for (const [index, message] of chat.messages.entries()) {
      const row = document.createElement("article");
      row.className = `message message-${message.role}`;
      row.dataset.messageId = message.id;
      row.setAttribute("aria-label", message.role === "user" ? "Your message" : "OserAI reply");
      const content = document.createElement("div"); content.className = "message-content";
      const body = document.createElement("div"); body.className = "message-body";
      if (message.role === "user") {
        body.textContent = message.content; row.append(body);
      } else {
        row.innerHTML = '<svg class="assistant-mark" aria-hidden="true"><use href="assets/icons.svg#mark"></use></svg>';
        const label = document.createElement("span"); label.className = "assistant-label"; label.textContent = "oserai";
        if (!message.content && message.state === "pending") body.innerHTML = '<div class="typing" aria-label="Preparing a reply"><span></span><span></span><span></span></div>';
        else body.innerHTML = markdown(message.content);
        content.append(label, body);
        if (["error", "interrupted"].includes(message.state)) {
          const error = document.createElement("p"); error.className = "message-error";
          error.textContent = message.error || "Reply stopped. You can try again when you're ready.";
          content.append(error);
        }
        const actions = document.createElement("div"); actions.className = "message-actions";
        if (message.content && message.state !== "pending") {
          const button = document.createElement("button"); button.type = "button";
          button.innerHTML = `${icon("copy")}<span>Copy</span>`;
          button.setAttribute("aria-label", "Copy reply"); button.addEventListener("click", () => copy(message.content, button)); actions.append(button);
        }
        if (["error", "interrupted"].includes(message.state) && index === chat.messages.length - 1) {
          const retry = document.createElement("button"); retry.innerHTML = `${icon("retry")}<span>Try again</span>`;
          retry.disabled = !ready || Boolean(generation);
          retry.addEventListener("click", () => {
            const user = chat.messages[index - 1];
            if (user?.role === "user") { chat.messages.splice(index - 1, 2); $("prompt").value = user.content; send(); }
          }); actions.append(retry);
        }
        if (actions.childNodes.length) content.append(actions);
        row.append(content);
      }
      $("messages").append(row);
    }
  }

  function controls() {
    const length = $("prompt").value.trim().length;
    $("send-button").disabled = !length || !ready || Boolean(generation) || length > maxInput;
    $("send-button").hidden = Boolean(generation);
    $("stop-button").hidden = !generation;
    $("character-count").textContent = `${$("prompt").value.length.toLocaleString()} / ${maxInput.toLocaleString()}`;
    $("character-count").hidden = length < maxInput * .8;
    $("prompt").style.height = "auto";
    $("prompt").style.height = `${Math.min($("prompt").scrollHeight, 180)}px`;
  }

  function render() { history(); renderMessages(); controls(); }
  function scrollDown() { const area = $("conversation-area"); area.scrollTop = area.scrollHeight; }

  function newChat() {
    stop(); currentId = null; $("prompt").value = ""; $("chat-search").value = "";
    render(); sidebar(false); $("prompt").focus();
  }

  function stop() { if (generation) generation.abort(); }

  async function availability() {
    const wasReady = ready;
    try {
      const response = await fetch("api.php?action=status", {credentials: "same-origin", cache: "no-store", signal: AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error("unavailable");
      const status = await response.json();
      ready = status.ready === true;
      csrf = typeof status.csrf_token === "string" ? status.csrf_token : "";
      maxInput = Math.max(1, Math.min(Number(status.max_input_chars) || 16000, 16000));
      $("prompt").maxLength = maxInput;
      $("model-name").textContent = status.model_label || "OserAI Chat";
    } catch { ready = false; }
    $("network-status").className = `network-status ${ready ? "ready" : "offline"}`;
    $("network-status").lastElementChild.textContent = ready ? "Ready to explore" : "Coming online";
    $("availability").hidden = ready;
    $("availability").textContent = ready ? "" : "Chat is coming online. Please check back soon.";
    controls();
    if (ready !== wasReady && !generation) renderMessages();
  }

  async function consume(response, onEvent) {
    if (!response.body) throw new Error("Streaming isn't available in this browser.");
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "", completed = false;
    try {
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value, {stream: !done});
        let boundary;
        while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          const lines = frame.split(/\r?\n/);
          const name = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
          const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
          if (!data) continue;
          const payload = JSON.parse(data);
          if (name === "error") throw new Error(payload.message || "The reply was interrupted. Please try again.");
          onEvent(name, payload);
          if (name === "done") completed = true;
        }
        if (done) break;
        if (buffer.length > 200000) throw new Error("The reply couldn't be read. Please try again.");
      }
      if (!completed) throw new Error("The connection ended before the reply was complete. Please try again.");
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  }

  async function send() {
    const text = $("prompt").value.trim();
    if (!text || !ready || generation || text.length > maxInput) return;
    let chat = current();
    if (!chat) {
      chat = {id: uid(), title: text.replace(/\s+/g, " ").slice(0, 55), updatedAt: Date.now(), messages: []};
      conversations.unshift(chat); conversations = conversations.slice(0, 30); currentId = chat.id;
    }
    const user = {id: uid(), role: "user", content: text, state: "complete", include: true};
    chat.messages.push(user);
    const context = chat.messages.filter((m) => m.include && m.state === "complete").map(({role, content}) => ({role, content}));
    while (context.length > 62 || (context.length > 1 && context.reduce((sum, m) => sum + m.content.length, 0) > 30000)) {
      context.shift(); if (context[0]?.role === "assistant") context.shift();
    }
    const assistant = {id: uid(), role: "assistant", content: "", state: "pending", include: false};
    chat.messages.push(assistant); chat.updatedAt = Date.now();
    conversations = [chat, ...conversations.filter((c) => c.id !== chat.id)];
    const controller = new AbortController(); generation = controller;
    $("prompt").value = ""; persist(); render(); scrollDown();
    $("screen-status").textContent = "OserAI is preparing a reply.";
    try {
      const response = await fetch("api.php?action=chat", {
        method: "POST", credentials: "same-origin", signal: controller.signal,
        headers: {"Content-Type": "application/json", "X-CSRF-Token": csrf},
        body: JSON.stringify({messages: context}),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        if (response.status === 403) availability();
        throw new Error(payload.message || "Chat is unavailable right now. Please try again shortly.");
      }
      await consume(response, (name, data) => {
        if (name === "token" && typeof data.text === "string") {
          assistant.content += data.text;
          if (currentId === chat.id) {
            const area = $("conversation-area"), nearBottom = area.scrollHeight - area.scrollTop - area.clientHeight < 140;
            renderMessages(); if (nearBottom) scrollDown();
          }
        }
        if (name === "done") {
          assistant.state = "complete"; assistant.include = true;
        }
      });
      if (!assistant.content.trim()) throw new Error("The reply was empty. Please try again.");
      $("screen-status").textContent = "Reply complete.";
    } catch (error) {
      assistant.state = controller.signal.aborted ? "interrupted" : "error";
      assistant.error = controller.signal.aborted ? "Reply stopped." : error.message;
      assistant.include = false; user.include = false;
      $("screen-status").textContent = assistant.error;
    } finally {
      if (generation === controller) generation = null;
      persist(); history();
      if (currentId === chat.id) renderMessages();
      controls();
    }
  }

  function confirmDelete(all, action) {
    $("confirm-dialog").querySelector("h2").textContent = all ? "Clear all conversations?" : "Clear this conversation?";
    $("confirm-dialog").querySelector("p").textContent = all ? "This removes every saved conversation from this browser." : "This removes the saved conversation from this browser.";
    $("confirm-delete").textContent = all ? "Clear all conversations" : "Delete conversation";
    deleteAction = action; $("confirm-dialog").showModal();
  }

  $("chat-form").addEventListener("submit", (event) => { event.preventDefault(); send(); });
  $("prompt").addEventListener("input", controls);
  $("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  });
  $("stop-button").addEventListener("click", stop);
  $("new-chat").addEventListener("click", newChat);
  $("chat-search").addEventListener("input", history);
  $("menu-button").addEventListener("click", () => sidebar(true));
  $("sidebar-scrim").addEventListener("click", () => sidebar(false));
  $("about-button").addEventListener("click", () => { sidebar(false); $("about-dialog").showModal(); });
  document.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
  $("clear-history").addEventListener("click", () => {
    $("about-dialog").close();
    confirmDelete(true, () => { stop(); conversations = []; currentId = null; persist(); render(); toast("Saved conversations cleared."); });
  });
  $("confirm-cancel").addEventListener("click", () => $("confirm-dialog").close());
  $("confirm-delete").addEventListener("click", () => { deleteAction?.(); $("confirm-dialog").close(); });
  document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => {
    $("prompt").value = button.dataset.prompt; controls(); $("prompt").focus();
  }));
  $("export-chat").addEventListener("click", () => {
    const chat = current(); if (!chat) return;
    const blob = new Blob([JSON.stringify({title: chat.title, messages: chat.messages.map(({role, content, state}) => ({role, content, state}))}, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `oserai-${chat.title.replace(/[^a-z0-9]+/gi, "-").slice(0, 45)}.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") sidebar(false);
    if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "o") { event.preventDefault(); newChat(); }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); if (innerWidth <= 760) sidebar(true); $("chat-search").focus(); }
  });
  load(); sidebar(false); render(); availability();
  window.addEventListener("resize", () => { if (innerWidth > 760) sidebar(false); else if (!$("sidebar").classList.contains("open")) $("sidebar").inert = true; });
  setInterval(() => { if (document.visibilityState === "visible") availability(); }, 30000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") availability(); });
})();
