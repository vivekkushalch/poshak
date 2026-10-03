import { api } from "./api.js";
import { $, errorText, h, progressBar, timeAgo } from "./kit.js";

const REFRESH_MS = 4000;
let timer = null;
let open = new Set();

function pill(job) {
  const kind = job.failed ? "bad" : job.status === "done" ? "ok" : "warn";
  const text = job.status === "done" ? (job.failed ? "Finished with errors" : "Done") : "Running";
  return h("span", { class: `pill ${kind}` }, text);
}

/** The API often prefixes the file name already; don't print it twice. */
function reason(e) {
  const msg = e.error || "unknown error";
  return msg.startsWith(`${e.photo_name} `) ? msg.slice(e.photo_name.length + 1) : msg;
}

function errorsList(batch) {
  const box = h("div", {}, h("span", { class: "muted" }, "Loading errors…"));
  api(`/index/jobs/${batch}`).then((r) => {
    if (!r.ok) return box.replaceChildren(h("span", { class: "error" }, errorText(r.body)));
    const errs = r.body.errors;
    box.replaceChildren(errs.length
      ? h("ul", {}, errs.map((e) => h("li", {}, h("code", {}, e.photo_name), " ", h("span", { class: "why" }, reason(e)))))
      : h("span", { class: "muted" }, "No errors."));
  }).catch((e) => box.replaceChildren(h("span", { class: "error" }, e.message)));
  return box;
}

function jobCard(j) {
  const details = h("details", {}, h("summary", {}, "Details and errors"));
  if (open.has(j.batch_id)) details.open = true;
  let filled = false;
  const fill = () => {
    if (details.open && !filled) { filled = true; details.append(errorsList(j.batch_id)); }
    if (details.open) open.add(j.batch_id); else open.delete(j.batch_id);
  };
  details.addEventListener("toggle", fill);
  fill();
  return h("article", { class: "card job" },
    h("div", { class: "job-top" },
      h("span", { class: "job-id", title: j.batch_id }, j.batch_id.length > 16 ? `${j.batch_id.slice(0, 8)}…${j.batch_id.slice(-4)}` : j.batch_id),
      h("span", { class: "row" }, pill(j), h("span", { class: "muted" }, timeAgo(j.updated_at)))),
    progressBar(j),
    h("div", { class: "counts" },
      h("span", {}, h("b", {}, j.done), " done"),
      h("span", {}, h("b", {}, j.skipped), " skipped"),
      h("span", {}, h("b", {}, j.failed), " failed"),
      h("span", {}, h("b", {}, j.total), " total"),
      j.api_count || j.local_count ? h("span", {}, `cutouts: ${j.api_count} remote, ${j.local_count} local`) : null),
    details);
}

export async function load() {
  try {
    const r = await api("/index/jobs");
    if (!r.ok) throw new Error(errorText(r.body));
    const jobs = r.body.jobs;
    const running = jobs.filter((j) => j.status !== "done").length;
    $("jobs-status").textContent = jobs.length
      ? `${jobs.length} batch${jobs.length === 1 ? "" : "es"}${running ? `, ${running} running` : ""}.`
      : "No indexing jobs yet. Upload photos to start one.";
    $("jobs-list").replaceChildren(...jobs.map(jobCard));
    if (!running) stop();
    else if (!timer) timer = setInterval(load, REFRESH_MS);
  } catch (e) {
    $("jobs-status").textContent = e.message;
    stop();
  }
}

function stop() {
  clearInterval(timer);
  timer = null;
}

export function init() { $("jobs-refresh").addEventListener("click", load); }
export const show = load;
export const hide = stop;
