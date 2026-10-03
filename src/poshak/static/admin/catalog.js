import { getImageBase, imgUrl } from "/static/common.js";
import { api, form } from "./api.js";
import { $, confirmDialog, errorText, h, openLightbox, toast } from "./kit.js";

const state = { page: 1, pageSize: 24, total: 0, items: [], filter: "", base: "", bust: new Map() };

function src(kind, fileName, key) {
  const url = imgUrl(state.base, kind, fileName);
  const v = state.bust.get(key);
  return v ? `${url}${url.includes("?") ? "&" : "?"}v=${v}` : url;
}

function thumb(label, url, name) {
  if (!url) {
    return h("figure", { class: "thumb" }, h("div", { class: "empty" }, "No cutout"), h("figcaption", {}, label));
  }
  const img = h("img", { src: url, alt: `${label} of ${name}`, loading: "lazy" });
  img.addEventListener("error", () => img.replaceWith(h("div", { class: "missing" }, "Image not found")));
  img.addEventListener("click", () => openLightbox(url, `${label}: ${name}`));
  return h("figure", { class: "thumb" }, img, h("figcaption", {}, label));
}

async function replace(it, kind, file, el) {
  if (!file) return;
  el.classList.add("busy");
  try {
    const r = await api("/replace", {
      method: "POST",
      body: form({ image: file, photo_name: it.photo_name, kind }),
    });
    if (!r.ok) throw new Error(errorText(r.body));
    state.bust.set(it.photo_name, Date.now());
    toast(`Replaced the ${kind === "rmbg" ? "cutout" : "photo"} for ${it.photo_name}`);
    await load();
  } catch (e) {
    el.classList.remove("busy");
    toast(e.message, "bad");
  }
}

async function remove(it, el) {
  const ok = await confirmDialog({
    title: "Delete this photo?",
    body: `${it.photo_name} (SKU ${it.sku_id}) will be removed along with its cutout, features and index entry. This can't be undone.`,
    action: "Delete",
    danger: true,
  });
  if (!ok) return;
  el.classList.add("busy");
  try {
    const r = await api("/forget", { method: "POST", body: form({ photo_name: it.photo_name }) });
    if (!r.ok) throw new Error(errorText(r.body));
    toast(`Deleted ${it.photo_name}`);
    if (state.items.length === 1 && state.page > 1) state.page -= 1;
    await load();
  } catch (e) {
    el.classList.remove("busy");
    toast(e.message, "bad");
  }
}

function picker(text, kind, it, el) {
  return h(
    "label",
    { class: "btn ghost sm" },
    text,
    h("input", { type: "file", accept: "image/*", hidden: true, onchange: (e) => replace(it, kind, e.target.files[0], el) }),
  );
}

function card(it) {
  const el = h("article", { class: "card item" });
  el.append(
    h("div", { class: "item-head" }, h("span", { class: "sku" }, `SKU ${it.sku_id}`), h("span", { class: "sub" }, it.item || "")),
    h(
      "div",
      { class: "pair" },
      thumb("Photo", src("uploads", it.photo_name, it.photo_name), it.photo_name),
      thumb("Cutout", it.rembg_name ? src("rmbg", it.rembg_name, it.photo_name) : "", it.photo_name),
    ),
    h("div", {},
      h("div", { class: "fname", title: it.photo_name }, it.photo_name),
      it.designno ? h("div", { class: "sub" }, `Design ${it.designno}`) : null),
    h(
      "div",
      { class: "actions" },
      picker("Replace photo", "original", it, el),
      it.rembg_name ? picker("Replace cutout", "rmbg", it, el) : null,
      h("button", { class: "btn danger-ghost sm", type: "button", onclick: () => remove(it, el) }, "Delete"),
    ),
  );
  return el;
}

function visibleItems() {
  const q = state.filter.trim().toLowerCase();
  if (!q) return state.items;
  return state.items.filter((it) =>
    `${it.sku_id} ${it.item || ""} ${it.designno || ""} ${it.photo_name}`.toLowerCase().includes(q));
}

function render() {
  const grid = $("grid");
  grid.replaceChildren();
  const items = visibleItems();
  if (!state.total) {
    grid.append(h("div", { class: "card empty-state" },
      h("strong", {}, "No photos indexed yet"),
      h("p", { class: "muted" }, "Upload a few photos and they will show up here once indexing finishes."),
      h("a", { class: "btn primary", href: "#upload" }, "Upload photos")));
  } else if (!items.length) {
    grid.append(h("div", { class: "card empty-state" }, h("strong", {}, "Nothing on this page matches the filter")));
  } else {
    items.forEach((it) => grid.append(card(it)));
  }
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $("pageinfo").textContent = `Page ${state.page} of ${pages}`;
  $("prev").disabled = state.page <= 1;
  $("next").disabled = state.page >= pages;
  $("status").textContent = state.total
    ? `${state.total} photo${state.total === 1 ? "" : "s"} indexed` +
      (state.filter ? `, ${items.length} shown on this page` : "")
    : "The index is empty.";
}

function skeletons() {
  const grid = $("grid");
  grid.replaceChildren(...Array.from({ length: 8 }, () => h("div", { class: "skeleton" })));
}

export async function load() {
  if (!state.items.length) skeletons();
  try {
    state.base = await getImageBase();
    const r = await api(`/skus?page=${state.page}&page_size=${state.pageSize}`);
    if (!r.ok) throw new Error(errorText(r.body));
    state.total = r.body.total;
    state.items = r.body.items;
    render();
    document.dispatchEvent(new Event("poshak:changed")); // keeps the header counts current
  } catch (e) {
    $("grid").replaceChildren(h("div", { class: "card empty-state" },
      h("strong", {}, "Couldn't load photos"), h("p", { class: "muted" }, e.message),
      h("button", { class: "btn primary", type: "button", onclick: load }, "Try again")));
  }
}

export function init() {
  $("prev").addEventListener("click", () => { state.page -= 1; load(); });
  $("next").addEventListener("click", () => { state.page += 1; load(); });
  $("filter").addEventListener("input", (e) => { state.filter = e.target.value; render(); });
  $("page-size").addEventListener("change", (e) => {
    state.pageSize = Number(e.target.value);
    state.page = 1;
    load();
  });
}

export const show = load;
export function hide() {}
