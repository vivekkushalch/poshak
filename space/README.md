---
title: poshak
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 8088
pinned: false
license: mit
short_description: Find the exact textile from a photo, not a lookalike
---

# poshak demo

Visual search for Indian wear that finds the exact piece, not just a lookalike. This Space runs
the real pipeline (SigLIP and SIFT, on CPU) over a gallery of 95 freely licensed textile and
garment photos. Pick a sample, or upload any photo.

The demo is read-only: you can search, but nothing can be added or changed.

- Code, docs and benchmarks: https://github.com/vivekkushalch/poshak
- Photo credits: `demo/ATTRIBUTION.md` (Wikimedia Commons; CC0, public domain or CC BY)
- The first request after the Space has been idle can take a minute while it wakes up.
