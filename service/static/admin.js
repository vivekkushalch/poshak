import { apiFetch, getImageBase, imgUrl } from "/static/common.js";

const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 50;
let page = 1, total = 0, base = "";

async function load() {
  base = await getImageBase();
  const res = await apiFetch(`/skus?page=${page}&page_size=${PAGE_SIZE}`);
  const body = await res.json();
  total = body.total;
  $("pageinfo").textContent =
    `page ${body.page} / ${Math.max(1, Math.ceil(total / PAGE_SIZE))} (${total} images)`;
  render(body.items);
}

function render(items) {
  const grid = $("grid");
  grid.innerHTML = "";
  items.forEach((it) => {
    const orig = imgUrl(base, "uploads", it.photo_name);
    const rmbg = it.rembg_name ? imgUrl(base, "rmbg", it.rembg_name) : "";
    const cell = document.createElement("div");
    cell.className = "cell card";
    cell.innerHTML = `
      <div><b>SKU ${it.sku_id}</b> — ${it.item || ""} <span class="muted">${it.designno || ""}</span></div>
      <div class="pair">
        <div><div class="muted">original</div><img loading="lazy" src="${orig}" alt="" /></div>
        <div><div class="muted">bg-removed</div>${rmbg ? `<img loading="lazy" src="${rmbg}" alt="" />` : '<div class="muted">—</div>'}</div>
      </div>
      <div class="muted">${it.photo_name}</div>
      <div class="row">
        <label class="secondary" style="padding:6px 10px;border-radius:8px;cursor:pointer">
          Replace original<input type="file" accept="image/*" hidden data-kind="original" data-photo="${it.photo_name}" />
        </label>
        ${it.rembg_name ? `<label class="secondary" style="padding:6px 10px;border-radius:8px;cursor:pointer">
          Replace bg<input type="file" accept="image/*" hidden data-kind="rmbg" data-photo="${it.photo_name}" /></label>` : ""}
      </div>`;
    cell.querySelectorAll('input[type=file]').forEach((inp) =>
      inp.addEventListener("change", () => doReplace(inp)));
    grid.appendChild(cell);
  });
}

async function doReplace(inp) {
  const f = inp.files[0];
  if (!f) return;
  $("status").textContent = `replacing ${inp.dataset.photo}…`;
  const fd = new FormData();
  fd.append("image", f);
  fd.append("photo_name", inp.dataset.photo);
  fd.append("kind", inp.dataset.kind);
  const res = await apiFetch("/replace", { method: "POST", body: fd });
  const body = await res.json();
  $("status").textContent = res.ok ? `replaced ${body.photo_name}` : (body.detail || "error");
  if (res.ok) load();   // refresh
}

$("prev").addEventListener("click", () => { if (page > 1) { page--; load(); } });
$("next").addEventListener("click", () => {
  if (page * PAGE_SIZE < total) { page++; load(); }
});

load();
