"""Thread pools for the search pipeline.

Two separate pools so nested submits don't queue behind each other:
  - PIPELINE_POOL: 2 workers — cloth's A and B run concurrently.
  - SIFT_POOL:     sized to CPU — stage 2 candidates run in parallel.

cv2 SIFT, cv2 imread, torch eval/no_grad, and numpy ops all release the
GIL during their C work, so threads give a real speedup here.
"""

import os
from concurrent.futures import ThreadPoolExecutor


def _n_sift_workers() -> int:
    # Each worker holds: a candidate image (~2MB after downsize) +
    # SIFT descriptors + the BFMatcher's distance matrix. 3 in-flight
    # is a memory-vs-throughput sweet spot for the 4GB RAM budget.
    # Bump to cpu_count if you've got memory to burn.
    return max(2, min(3, os.cpu_count() or 3))


PIPELINE_POOL = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="search-pipe"
)
SIFT_POOL = ThreadPoolExecutor(
    max_workers=_n_sift_workers(), thread_name_prefix="search-sift"
)
