"""Service configuration.

Mirrors what the prototype (sharwani_large.ipynb) uses verbatim — the
model id, paths, item routing — plus the few service-level knobs the
notebook doesn't have (host/port, gc cadence).
"""

import os
from pathlib import Path


# --- model identity --------------------------------------------------------
# Locked to what the prototype loads:
#   processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
#   model     = AutoModel.from_pretrained("google/siglip-base-patch16-224")
# Bump MODEL_TAG whenever the model or preprocessing changes; rows in
# stock_image_vectors carry the same tag so the service refuses to serve
# vectors produced under a different model.
MODEL_HF_ID = "google/siglip-base-patch16-224"
MODEL_TAG = "siglip-base-patch16-224-v2-768"
EMBED_DIM = 768


# --- service ---------------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8088"))


# --- corpus paths ----------------------------------------------------------
# In docker-compose, c:/a local PHP server/htdocs/poshak/uploads is bind-mounted here RO.
# data/ holds bg-removed cloth PNGs produced by the bg_remove batch script.
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", "/corpus/uploads"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/corpus/data"))

# Pickle + .npy cache used by index build/load (prototype's save_cache /
# load_cache). Lives inside the container's writable volume.
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))


# --- item routing (verbatim from prototype) --------------------------------
# Prototype:
#   CLOTH_ITEMS  = {'Sherwani','Indowestern','Bandhgala','Tuxedo','Jacket','Bomber Jacket'}
#   FABRIC_ITEMS = {'Fabric'}
CLOTH_ITEMS = {"Sherwani", "Indowestern", "Bandhgala", "Tuxedo", "Jacket", "Bomber Jacket"}
FABRIC_ITEMS = {"Fabric"}


# --- working resolution (matches spipeline.py load_image max_h=768) --------
WORKING_MAX_H = int(os.environ.get("WORKING_MAX_H", "768"))

# --- local data (SQLite index + precomputed descriptors + change ledger) ---
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", str(DATA_ROOT / "index.sqlite")))
DESC_DIR = Path(os.environ.get("DESC_DIR", str(DATA_ROOT / "descriptors")))
LEDGER_PATH = Path(os.environ.get("LEDGER_PATH", str(DATA_ROOT / "pending_sync.jsonl")))

# --- corpus (local-first; seeded from S3) ----------------------------------
RMBG_DIR = Path(os.environ.get("RMBG_DIR", "/corpus/rmbg"))

# --- S3 cold-start source --------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")
S3_UPLOADS_PREFIX = os.environ.get("S3_UPLOADS_PREFIX", "uploads/")
S3_RMBG_PREFIX = os.environ.get("S3_RMBG_PREFIX", "rmbg/")

# --- frontend display ------------------------------------------------------
# Public base; frontends build f"{IMAGE_BASE_URL}/uploads/<file>" etc.
IMAGE_BASE_URL = os.environ.get("IMAGE_BASE_URL", "")


# --- memory management -----------------------------------------------------
# Periodic gc + cuda cache flush keeps RSS flat over thousands of calls.
GC_EVERY_N_REQUESTS = int(os.environ.get("GC_EVERY", "50"))


# --- auth ------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "")
DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "0") == "1"
