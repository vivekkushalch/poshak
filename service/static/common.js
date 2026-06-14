// Shared helpers for the test + admin pages.
const KEY_NAME = "poshak_api_key";

export function getApiKey() {
  let k = localStorage.getItem(KEY_NAME);
  if (k === null) {
    k = prompt("Enter API key (leave blank if auth disabled):") || "";
    localStorage.setItem(KEY_NAME, k);
  }
  return k;
}

export function clearApiKey() {
  localStorage.removeItem(KEY_NAME);
}

export async function apiFetch(path, opts = {}) {
  const headers = Object.assign({ "X-API-Key": getApiKey() }, opts.headers || {});
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    clearApiKey();
    throw new Error("Unauthorized — API key cleared, reload to re-enter.");
  }
  return res;
}

let _imgBase = null;
export async function getImageBase() {
  if (_imgBase === null) {
    const cfg = await (await fetch("/config")).json();
    _imgBase = cfg.image_base_url || "";
  }
  return _imgBase;
}

export function imgUrl(base, kind, name) {
  // kind: 'uploads' | 'rmbg'
  return `${base}/${kind}/${name}`;
}
