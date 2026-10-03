import { api, form } from "./api.js";
import { $, errorText, formatBytes, h, newBatchId, progressBar, toast } from "./kit.js";

const CONCURRENCY = 3;
const POLL_MS = 1500;
const state = { files: [], busy: false };

const keyOf = (f) => `${f.name}:${f.size}`;

function addFiles(list) {
  for (const f of list) {
    if (f.type.startsWith("image/") && !state.files.some((x) => keyOf(x.file) === keyOf(f))) {
      state.files.push({ file: f, state: "waiting", note: "" });
    }
  }
  renderList();
}

function renderList() {
  const ul = $("up-list");
  ul.replaceChildren(
    ...state.files.map((entry, i) =>
      h("li", {},
        h("span", { class: "name", title: entry.file.name }, entry.file.name),
        entry.state === "waiting" && !state.busy
          ? h("button", { class: "x", type: "button", "aria-label": `Remove ${entry.file.name}`, onclick: () => { state.files.splice(i, 1); renderList(); } }, "✕")
          : h("span", { class: `state ${entry.state === "queued" ? "ok" : entry.state === "failed" ? "bad" : ""}` }, entry.state),
        h("span", { class: "meta" }, `${formatBytes(entry.file.size)}${entry.note ? ` · ${entry.note}` : ""}`))),
  );
  $("up-count").textContent = state.files.length;
}

function showError(msg) {
  const el = $("up-error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

const photoName = (file, sku) => {
  const clean = file.name.replace(/[\\/]/g, "_").replace(/\.{2,}/g, ".");
  return $("up-prefix").checked ? `${sku}_${clean}` : clean;
};

async function pool(items, n, fn) {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) await fn(items[next++]);
  }));
}

function setProgress(job, summary) {
  $("up-progress-wrap").hidden = false;
  $("up-progress").replaceChildren(progressBar(job));
  $("up-summary").replaceChildren(summary);
}

async function waitForJob(batch) {
  for (;;) {
    const r = await api(`/index/jobs/${batch}`);
    if (!r.ok) throw new Error(errorText(r.body));
    const j = r.body;
    const finished = j.done + j.failed + j.skipped;
    setProgress(j, `Indexed ${finished} of ${j.total}: ${j.done} done, ${j.skipped} skipped, ${j.failed} failed.`);
    if (j.status === "done") return j;
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

async function submit(event) {
  event.preventDefault();
  if (state.busy) return;
  const sku = Number($("up-sku").value);
  const item = $("up-item").value.trim();
  if (!state.files.length) return showError("Add at least one photo.");
  if ($("up-sku").value === "" || !Number.isInteger(sku) || sku < 0) return showError("Enter the SKU id as a whole number.");
  if (!item) return showError("Enter the item type, for example Sherwani.");
  showError("");

  state.busy = true;
  $("up-go").disabled = true;
  const batch = newBatchId();
  const queue = state.files.filter((e) => e.state === "waiting" || e.state === "failed");
  let queued = 0;
  setProgress({ total: 1, done: 0, skipped: 0, failed: 0 }, "Uploading…");
  await pool(queue, CONCURRENCY, async (entry) => {
    entry.state = "uploading";
    renderList();
    try {
      const r = await api("/ingest", {
        method: "POST",
        body: form({
          image: entry.file,
          sku_id: sku,
          item,
          photo_name: photoName(entry.file, sku),
          designno: $("up-design").value.trim(),
          batch_id: batch,
          bg_engine: $("up-engine").value,
          force: $("up-force").checked,
        }),
      });
      if (!r.ok) throw new Error(errorText(r.body));
      entry.state = "queued";
      queued += 1;
    } catch (e) {
      entry.state = "failed";
      entry.note = e.message;
    }
    renderList();
  });

  try {
    if (queued) {
      const job = await waitForJob(batch);
      toast(job.failed ? `${job.done} indexed, ${job.failed} failed. See the Jobs tab.` : `Indexed ${job.done + job.skipped} photo${job.done + job.skipped === 1 ? "" : "s"}.`, job.failed ? "bad" : "ok");
    } else {
      setProgress({ total: 1, done: 0, skipped: 0, failed: 1 }, "No photos were accepted.");
    }
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    document.dispatchEvent(new Event("poshak:changed"));
    state.busy = false;
    $("up-go").disabled = false;
    renderList();
  }
}

async function loadItemTypes() {
  try {
    const r = await api("/categories");
    if (r.ok) {
      $("item-list").replaceChildren(...r.body.categories.map((c) => h("option", { value: c.item })));
    }
  } catch {
    // the datalist is a convenience only
  }
}

export function init() {
  const dz = $("dropzone");
  $("up-files").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); });
  dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  $("up-form").addEventListener("submit", submit);
  $("up-clear").addEventListener("click", () => {
    if (state.busy) return;
    state.files = [];
    $("up-progress-wrap").hidden = true;
    showError("");
    renderList();
  });
  renderList();
}

export function show() { loadItemTypes(); }
export function hide() {}
