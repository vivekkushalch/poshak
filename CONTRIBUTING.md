# Contributing

Issues and pull requests are welcome.

## Setup

```bash
uv sync
uv run pytest -m "not slow"
uv run ruff check && uv run ruff format --check
```

CI runs the same three commands on every push and pull request.

## Ground rules

- Keep the layers apart. Routes call services, services call ports, and only
  `container.py` names concrete adapters. [docs/architecture.md](docs/architecture.md)
  has the full picture.
- A new backend (storage, embedder, vector index) is one class that satisfies a
  `Protocol` in `domain/ports.py`, a branch in `container.py`, and a test. Storage
  backends must pass `tests/integration/test_blobstore_contract.py`.
- Don't regenerate `tests/golden/search_golden.json` to make a change pass. It
  exists to catch ranking changes. If you mean to change ranking, regenerate it
  with `GOLDEN_WRITE=1 uv run pytest tests/unit/engine/test_pipeline.py`, explain why in the PR, and include before/after numbers.
- Scoring constants and SIFT parameters were tuned together
  ([ADR 0001](docs/adr/0001-sift-verification.md)). Change them only with an
  evaluation to back it.
- Conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
