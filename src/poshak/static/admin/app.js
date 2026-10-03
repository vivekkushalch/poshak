// Admin shell: hash router, health badge, API key dialog.
import { currentApiKey, hasApiKey, setApiKey } from "/static/common.js";
import * as catalog from "./catalog.js";
import * as categories from "./categories.js";
import * as jobs from "./jobs.js";
import { $ } from "./kit.js";
import * as upload from "./upload.js";

const tabs = { catalog, upload, jobs, categories };
let current = null;
// true once a non-empty key has been stored, so we can tell "rejected" from "never entered"
let keyEntered = hasApiKey() && currentApiKey() !== "";

function route() {
  const name = tabs[location.hash.slice(1)] ? location.hash.slice(1) : "catalog";
  if (name === current) return;
  if (current) tabs[current].hide();
  current = name;
  document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  document.querySelectorAll("[data-tab]").forEach((a) => {
    if (a.dataset.tab === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `${name[0].toUpperCase()}${name.slice(1)} · poshak admin`;
  tabs[name].show();
}

async function refreshHealth() {
  const el = $("health");
  try {
    const res = await fetch("/health");
    const b = await res.json();
    const n = b.index.cloth_orig + b.index.fabric;
    el.className = `pill ${b.index_ready ? "ok" : "warn"}`;
    el.textContent = b.index_ready ? `${n} photos indexed` : "Index empty";
    el.title = `Model ${b.model_tag}. Cloth ${b.index.cloth_orig} (${b.index.cloth_bg} cutouts), fabric ${b.index.fabric}.`;
  } catch {
    el.className = "pill bad";
    el.textContent = "Service unreachable";
    el.title = "";
  }
}

function askForKey(message = "") {
  const dlg = $("keydlg");
  if (dlg.open) return;
  $("key-input").value = currentApiKey();
  $("key-msg").textContent = message;
  $("key-msg").hidden = !message;
  dlg.returnValue = "";
  dlg.addEventListener("close", () => {
    if (dlg.returnValue === "save") {
      const key = $("key-input").value.trim();
      keyEntered = key !== "";
      setApiKey(key);
      tabs[current].show();
    }
  }, { once: true });
  dlg.showModal();
}

function boot() {
  Object.values(tabs).forEach((t) => t.init());
  $("key-btn").addEventListener("click", () => askForKey());
  $("lightbox").addEventListener("click", () => $("lightbox").close());
  document.addEventListener("poshak:unauthorized", () =>
    askForKey(keyEntered ? "That key was rejected. Enter the current API key." : "This server needs an API key."));
  document.addEventListener("poshak:changed", refreshHealth);
  window.addEventListener("hashchange", route);
  if (!hasApiKey()) setApiKey(""); // an empty key works when authentication is off
  route();
  refreshHealth();
  setInterval(refreshHealth, 20000);
}

boot();
