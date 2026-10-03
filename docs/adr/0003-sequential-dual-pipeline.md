# 3. Run the two cloth passes one after the other

## Context

A cloth search runs a cutout pass and an original pass. Running them in parallel
would cut latency but holds both passes' images and descriptors at once, roughly
doubling peak memory. The earlier deployment ran them sequentially to stay clear
of the OOM killer, and the 2 GB target is tighter still.

## Decision

Run the passes sequentially. Inside each pass, verify the 40 candidates on a shared
thread pool. OpenCV releases the GIL during matching, so threads give real
parallelism there without the memory of two full passes.

## Consequences

- Latency is the sum of two passes, minus what the verification pool saves.
- On a bigger machine, running the passes concurrently is a small change in
  `engine/pipeline.cloth_search`.
