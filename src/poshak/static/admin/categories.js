import { api, form } from "./api.js";
import { $, errorText, h, toast } from "./kit.js";

async function save(item, category) {
  const r = await api("/categories", { method: "POST", body: form({ item, category }) });
  if (!r.ok) throw new Error(errorText(r.body));
  toast(`${r.body.item} now uses the ${r.body.category} pipeline`);
}

function row(c) {
  const select = h("select", { "aria-label": `Pipeline for ${c.item}` },
    h("option", { value: "cloth", selected: c.category === "cloth" }, "cloth (photo + cutout)"),
    h("option", { value: "fabric", selected: c.category === "fabric" }, "fabric (photo only)"));
  select.addEventListener("change", async () => {
    try { await save(c.item, select.value); await load(); } catch (e) { toast(e.message, "bad"); select.value = c.category; }
  });
  return h("tr", {}, h("td", {}, c.item), h("td", {}, h("span", { class: `src ${c.source}` }, c.source)), h("td", {}, select));
}

export async function load() {
  try {
    const r = await api("/categories");
    if (!r.ok) throw new Error(errorText(r.body));
    $("cat-body").replaceChildren(...r.body.categories.map(row));
  } catch (e) {
    $("cat-body").replaceChildren(h("tr", {}, h("td", { colspan: "3", class: "error" }, e.message)));
  }
}

export function init() {
  $("cat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const item = $("cat-item").value.trim();
    if (!item) return toast("Enter an item type first.", "bad");
    try {
      await save(item, $("cat-category").value);
      $("cat-item").value = "";
      await load();
    } catch (err) {
      toast(err.message, "bad");
    }
  });
}

export const show = load;
export function hide() {}
