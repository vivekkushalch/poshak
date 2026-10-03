import { apiFetch } from "/static/common.js";

/** JSON request helper. Returns {ok, status, body}. Announces 401s so the shell can ask for a key. */
export async function api(path, opts) {
  let res;
  try {
    res = await apiFetch(path, opts);
  } catch (err) {
    if (err.status === 401) document.dispatchEvent(new Event("poshak:unauthorized"));
    throw err;
  }
  let body = {};
  try {
    body = await res.json();
  } catch {
    // empty or non-JSON body
  }
  return { ok: res.ok, status: res.status, body };
}

export function form(fields) {
  const fd = new FormData();
  for (const [key, value] of Object.entries(fields)) fd.append(key, value);
  return fd;
}
