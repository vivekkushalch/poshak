import { apiFetch, getImageBase, imgUrl } from "/static/common.js";

const $ = (id) => document.getElementById(id);
const results = $("results");
const statusEl = $("status");

function openLightbox(src) {
  $("lightbox-img").src = src;
  $("lightbox").classList.add("open");
}
$("lightbox").addEventListener("click", () => $("lightbox").classList.remove("open"));

function renderResults(items, base) {
  results.innerHTML = "";
  items.forEach((r, i) => {
    const e = r.best_entry || {};
    const url = imgUrl(base, "uploads", e.json_filename || e.filename);
    const score = (r.unified_score ?? 0).toFixed(3);
    const tier = r.tier || r.orig_tier || r.bg_tier || "";
    const div = document.createElement("div");
    div.className = "result card" + (i === 0 ? " top" : "");
    div.innerHTML = `
      <img loading="lazy" src="${url}" alt="${e.filename || ""}" />
      <div>
        <div><b>#${i + 1}</b> SKU ${r.sku_id} — ${e.item || ""} ${i === 0 ? "★" : ""}</div>
        <div class="score">confidence ${score} ${tier}</div>
        <div class="muted">SIFT matches: ${r.total_sift_matches ?? 0}</div>
        <div class="muted">file: ${e.filename || ""}</div>
      </div>`;
    div.querySelector("img").addEventListener("click", () => openLightbox(url));
    results.appendChild(div);
  });
}

$("go").addEventListener("click", async () => {
  const f = $("file").files[0];
  if (!f) { statusEl.textContent = "pick or capture an image first"; return; }
  statusEl.textContent = "searching…";
  const base = await getImageBase();
  const fd = new FormData();
  fd.append("image", f);
  fd.append("category", $("category").value);
  fd.append("top_k", $("topk").value);
  try {
    const res = await apiFetch("/search", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) { statusEl.textContent = body.detail || "error"; return; }
    statusEl.textContent = `${body.total_found} results`;
    renderResults(body.results, base);
  } catch (e) {
    statusEl.textContent = e.message;
  }
});
