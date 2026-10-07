# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting:
the **Security** tab of this repository, then **Report a vulnerability**. Include what you
found, how to reproduce it and which version or commit you tested. You will get a reply
within a few days.

## Supported versions

Only the latest release and `main` receive fixes.

## What to know when you deploy poshak

- Data routes require an `X-API-Key` header, compared in constant time. The service refuses
  to start without `API_KEY` unless `DISABLE_AUTH=1`, which is meant for local development only.
- `/img/*`, `/ui`, `/admin`, `/static` and the `/health*` probes are served without a key.
  The admin page's data calls still need one. Put the service behind a reverse proxy with TLS,
  and restrict `/img/*` there if your photos are not public.
- Uploaded photos are decoded with OpenCV. Keep the container image and its dependencies
  up to date.
- The container runs as an unprivileged user (uid 1000) and keeps all state under `/data`
  and `/corpus`.
