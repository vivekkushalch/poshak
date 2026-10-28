import { currentApiKey, getImageBase, imgUrl } from "/static/common.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let base = "";
let queryUrl = null;

/** Create an element. Text goes in as text nodes, so catalog data can't inject markup. */
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return node;
}

function openLightbox(src, caption) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = caption;
  $("lightbox-cap").textContent = caption;
  $("lightbox").showModal();
}
$("lightbox").addEventListener("click", () => $("lightbox").close());

function renderResults(items) {
  const top = Math.max(1, ...items.map((r) => r.total_sift_matches ?? 0));
  $("results").replaceChildren(
    ...items.map((r, i) => {
      const e = r.best_entry || {};
      const name = e.json_filename || e.filename;
      const url = imgUrl(base, "uploads", name);
      const matches = r.total_sift_matches ?? 0;
      return el(
        "article",
        { class: "ui-res" + (i === 0 ? " top" : ""), style: `--i:${i}` },
        el("button", { class: "ui-img", type: "button", "aria-label": `Open the photo for SKU ${r.sku_id}`,
          onclick: () => openLightbox(url, `SKU ${r.sku_id}${e.item ? ", " + e.item : ""}`) },
          el("img", { src: url, alt: `Catalog photo for SKU ${r.sku_id}`, loading: "lazy" })),
        el("div", { class: "ui-body" },
          el("div", { class: "ui-rank" }, el("span", { class: "n" }, `#${i + 1}`), `SKU ${r.sku_id}`),
          e.item ? el("div", { class: "muted" }, e.item) : null,
          el("div", { class: "ui-meter", role: "img", "aria-label": `${matches} matching keypoints` },
            el("i", { style: `width:${Math.max(3, (matches / top) * 100)}%` })),
          el("div", { class: "muted" }, `${matches} matching keypoint${matches === 1 ? "" : "s"}`)),
      );
    }),
  );
}

async function search(file) {
  statusEl.textContent = "Searching…";
  if (queryUrl) URL.revokeObjectURL(queryUrl);
  queryUrl = URL.createObjectURL(file);
  $("qprev").src = queryUrl;
  const fd = new FormData();
  fd.append("image", file);
  fd.append("category", $("category").value);
  fd.append("top_k", $("topk").value);
  const key = currentApiKey();
  const t0 = performance.now();
  try {
    const res = await fetch("/search", { method: "POST", body: fd, headers: key ? { "X-API-Key": key } : {} });
    const body = await res.json().catch(() => ({}));
    if (res.status === 401) { statusEl.textContent = "This server needs an API key. Set one on the Admin page."; return; }
    if (!res.ok) { statusEl.textContent = body.detail || `Search failed (${res.status}).`; return; }
    const secs = ((performance.now() - t0) / 1000).toFixed(2);
    statusEl.textContent = body.total_found ? "" : "No results. Is anything indexed yet?";
    $("meta").textContent = `${body.total_found} result${body.total_found === 1 ? "" : "s"} in ${secs} s, measured in your browser.`;
    $("out").hidden = false;
    renderResults(body.results);
    $("out").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    statusEl.textContent = e.message;
  }
}

$("go").addEventListener("click", () => {
  const f = $("file").files[0];
  if (!f) { statusEl.textContent = "Pick or take a photo first."; return; }
  search(f);
});

async function loadSamples() {
  try {
    const res = await fetch("/samples/index.json");
    if (!res.ok) return;
    const samples = await res.json();
    if (!samples.length) return;
    $("samples").replaceChildren(
      ...samples.map((s) =>
        el("button", { class: "ui-sample", type: "button", "aria-label": `Search with ${s.label}`,
          onclick: async () => {
            const blob = await (await fetch(`/samples/${encodeURIComponent(s.file)}`)).blob();
            if (s.category) $("category").value = s.category;
            search(new File([blob], s.file, { type: blob.type || "image/jpeg" }));
          } },
          el("img", { src: `/samples/${encodeURIComponent(s.file)}`, alt: s.label }),
          el("span", {}, s.label)),
      ),
    );
    $("samplesBox").hidden = false;
    $("foot").hidden = false;
    $("ownTitle").textContent = "Or search with your own photo";
  } catch {
    // samples are optional
  }
}

async function loadHealth() {
  const pill = $("health");
  try {
    const h = await (await fetch("/health")).json();
    const n = h.index.cloth_orig + h.index.fabric;
    pill.className = `pill ${h.index_ready ? "ok" : "warn"}`;
    pill.textContent = h.index_ready ? `${n} photos indexed` : "Index empty";
    pill.title = `Ranker: ${Object.entries(h.ranker_active || {}).map(([k, v]) => `${k} ${v}`).join(", ")}`;
  } catch {
    pill.className = "pill bad";
    pill.textContent = "Service unreachable";
  }
}

getImageBase().then((b) => { base = b; });
loadSamples();
loadHealth();
