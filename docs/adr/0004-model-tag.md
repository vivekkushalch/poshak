# 4. Stamp every index row with a model tag

## Context

Embeddings from two models, or from one model at two resolutions, aren't
comparable. A partially re-indexed catalog would return confident nonsense.

## Decision

Every row stores `model_tag`, derived from the embedding model id and the working
resolution. The index loads only rows whose tag matches the running configuration.
The default model keeps its historical tag, `siglip-base-patch16-224-v2-768`, so
existing indexes keep loading.

## Consequences

- Changing the model is safe: the old rows go dark until re-indexed with
  `poshak seed --force`.
- Search returns fewer results during a re-index, never wrong ones.
