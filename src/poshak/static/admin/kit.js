// Small DOM helpers shared by every admin tab. No framework, no build step.

export const $ = (id) => document.getElementById(id);

/** Create an element. Text children go in as text nodes, so data can't inject markup. */
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value == null) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function toast(message, kind = "ok") {
  const el = h("div", { class: `toast ${kind}`, role: kind === "bad" ? "alert" : "status" }, message);
  $("toasts").append(el);
  setTimeout(() => el.classList.add("out"), 3800);
  setTimeout(() => el.remove(), 4200);
}

/** Resolves true when the user confirms. */
export function confirmDialog({ title, body, action = "Confirm", danger = false }) {
  const dlg = $("confirm");
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  const ok = $("confirm-ok");
  ok.textContent = action;
  ok.className = danger ? "btn danger" : "btn primary";
  return new Promise((resolve) => {
    dlg.returnValue = "";
    dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
    dlg.showModal();
  });
}

export function openLightbox(src, caption) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = caption;
  $("lightbox-cap").textContent = caption;
  $("lightbox").showModal();
}

/** Turn an API error body (string or FastAPI validation list) into one line. */
export function errorText(body, fallback = "Request failed") {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  }
  return fallback;
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** SQLite stores UTC as "YYYY-MM-DD HH:MM:SS". */
export function timeAgo(sqliteUtc) {
  const then = new Date(sqliteUtc.replace(" ", "T") + "Z").getTime();
  const secs = Math.round((then - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of [["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(secs) >= size) return rtf.format(Math.round(secs / size), unit);
  }
  return rtf.format(secs, "second");
}

export function newBatchId() {
  return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, "");
}

/** Two-segment progress bar: successes in the accent colour, failures in red. */
export function progressBar(job) {
  const total = job.total || 1;
  const ok = Math.round(((job.done + job.skipped) / total) * 100);
  const bad = Math.round((job.failed / total) * 100);
  const el = h("div", { class: "progress", role: "progressbar", "aria-label": "Indexing progress", "aria-valuemin": "0", "aria-valuemax": "100" },
    h("div", { class: "ok", style: `width:${ok}%` }), h("div", { class: "bad", style: `width:${bad}%` }));
  el.setAttribute("aria-valuenow", String(ok + bad));
  return el;
}
