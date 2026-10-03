# 5. Ports and adapters around a pure engine

## Context

The first version was a flat package where search, storage and model loading
reached into each other through module globals. Tests patched those globals, and
moving photos to S3 would have meant editing every file that opened a path.

## Decision

- The algorithm lives in `engine/` as pure functions.
- Storage, embedding, background removal, the vector index and persistence are
  `Protocol` ports in `domain/ports.py`, with implementations in `infrastructure/`.
- Services hold the use cases and receive their dependencies.
- `container.py` is the only place that chooses implementations.

## Consequences

- S3 support is one class plus one branch in the container. A contract test runs
  the same cases against local disk and S3.
- Tests build the real container with in-memory fakes instead of patching.
- There are more files than the line count strictly needs. Each one has one job,
  which is the trade this project makes on purpose.
