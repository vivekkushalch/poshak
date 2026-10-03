"""Stdlib logging setup and a timing helper for pipeline stages."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger("poshak")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


@contextmanager
def timed(label: str, logger: logging.Logger = log) -> Iterator[None]:
    """Log how long a block took, at DEBUG."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug("%s: %.0fms", label, (time.perf_counter() - t0) * 1000.0)
