"""poshak: fine-grained visual search for Indian wear."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("poshak")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
