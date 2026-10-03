# 2. Process images at 768 px height

## Context

Catalog photos are often 3000 px or more. SIFT on full resolution produces more
keypoints than the match budget uses and costs memory the 2 GB target doesn't have.

## Decision

Downscale every image by height to 768 px with `INTER_AREA` before embedding and
SIFT. Never upscale or crop. Background removal is the exception: it runs on the
full-size image, where U2Net has the most detail to segment, and its output is then
downscaled like everything else.

## Consequences

- Keypoint counts and match counts stay in the range the scoring tiers expect.
- The working resolution is part of the model tag, so changing it starts a new index
  instead of mixing incompatible descriptors.
