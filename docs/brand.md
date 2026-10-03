# poshak brand

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/poshak-logo-dark.svg">
  <img alt="poshak" src="assets/brand/poshak-logo-light.svg" width="320">
</picture>

## The mark

The mark is a zari lozenge, the diamond buti found on brocade borders and
sherwani yokes. It also draws the algorithm:

- the indigo field is the garment
- the gold ring is the zari border the search is built to notice
- the five gold dots are keypoints
- the thin lines between them are the matches SIFT verifies

<img alt="poshak mark" src="assets/brand/poshak-mark-light.svg" width="96">

Below 24 px, use `favicon.svg`. It drops the ring and the match lines and keeps the
field and the five keypoints, which stay legible at 16 px.

## Lockups

| Asset | Light background | Dark background |
|---|---|---|
| Horizontal (default) | `poshak-logo-light.svg` | `poshak-logo-dark.svg` |
| Stacked (square spaces, avatars) | `poshak-logo-stacked-light.svg` | `poshak-logo-stacked-dark.svg` |
| Mark only | `poshak-mark-light.svg`, `.png` (512 px) | `poshak-mark-dark.svg` |
| Favicon | `favicon.svg` | `favicon.svg` |
| README banner (1280×400) | `banner-light.svg` | `banner-dark.svg` |
| Social card (1280×640) | | `social-preview.png` |

All files are in [`docs/assets/brand`](assets/brand). Use the PNG only where SVG
isn't accepted. The social card is the image to upload under the repository's
Settings, then Social preview.

Leave clear space around a lockup of at least a quarter of the mark's height. Don't
recolour, rotate, stretch, outline or add effects to the mark, and don't set the
wordmark in another typeface.

## Colour

| Role | Light | Dark | Use |
|---|---|---|---|
| Indigo (field) | `#1e1b4b` | `#2e2a7a` | Mark field, wordmark on light |
| Zari gold | `#c9961a` | `#e8b84a` | Ring, keypoints, accents. Never body text |
| Ivory (paper) | `#f7f5ef` | | Light backgrounds |
| Night (ink) | | `#13112b` | Dark backgrounds |
| Moonlight | | `#f4f1ff` | Wordmark and text on dark |

Gold carries meaning, so use it sparingly. One gold element per view is the rule
the mark itself follows.

Charts don't use the brand colours. They use a categorical palette checked for
colour-blind separation and contrast in both themes (blue `#2a78d6`/`#3987e5`,
orange `#eb6834`/`#d95926`).

## Type

The wordmark is set in [Sora](https://github.com/sora-xor/sora-font) Bold, tracked
in by 3.5%, and converted to outlines, so it renders the same everywhere. Sora is
licensed under the SIL Open Font License 1.1. Copyright 2019 The Sora Project
Authors.

Product and documentation text use the platform's system sans-serif stack. The
wordmark is always lowercase: write the name as "poshak" in running text too.

## Voice

Plain and specific. Name the mechanism or the number instead of the feeling:
"verifies 40 candidates with SIFT" rather than "blazing-fast accurate search". The
tagline is the model for tone:

> SigLIP sees the silhouette. SIFT sees the zari.
