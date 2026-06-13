"""SigLIP singleton — matches the prototype's load exactly.

Prototype (sharwani_large.ipynb):
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model     = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE).eval()

    def siglip_embed(img_rgb):
        inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            ...
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

Loaded ONCE at process startup (module import). Loading per-request makes
each call 10-30s.
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from . import config
from .debug import timed


device = "cuda" if torch.cuda.is_available() else "cpu"

# Halve torch's intra-op threads so two parallel siglip_embed calls
# (cloth pipelines A and B) don't oversubscribe CPU. Without this,
# each call uses ALL cores and the parallel calls thrash on the
# scheduler, making the total slower than serial.
_torch_threads = max(1, (os.cpu_count() or 4) // 2)
torch.set_num_threads(_torch_threads)

processor = AutoProcessor.from_pretrained(config.MODEL_HF_ID)
model = AutoModel.from_pretrained(config.MODEL_HF_ID).to(device).eval()


def siglip_embed(img_rgb: np.ndarray) -> np.ndarray:
    """Verbatim port of the prototype's `siglip_embed`.

    `img_rgb`: uint8 RGB ndarray (same shape the prototype's load_image
    produces via cv2 BGR→RGB). transformers' image processor accepts
    numpy arrays directly.

    Returns: 768-d L2-normalized float numpy vector.

    The hasattr branching mirrors the prototype: depending on the
    transformers version, `get_image_features` returns either a raw
    Tensor or a `BaseModelOutputWithPooling`. We handle both.
    """
    with timed("siglip_embed"):
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state[:, 0, :]
            else:
                emb = out
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].detach().cpu().numpy().flatten()
