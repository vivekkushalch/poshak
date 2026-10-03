"""SigLIP image embedder. Port of the prototype's siglip_embed.

Loaded lazily on first use, then shared. The text tower is dropped after load
(only get_image_features is used), which frees about half the model's RAM."""

import gc
import os
import threading

import numpy as np

from ...core.logging import timed


class SiglipEmbedder:
    dim = 768

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._device = "cpu"

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModel, AutoProcessor

                # Half the cores per call so concurrent embeds don't thrash the scheduler.
                torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
                if hasattr(model, "text_model"):
                    model.text_model = None
                gc.collect()
                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = model
        return self._processor, self._model

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        import torch

        processor, model = self._load() if self._model is None else (self._processor, self._model)
        with timed("siglip_embed"), torch.no_grad():
            inputs = processor(images=img_rgb, return_tensors="pt").to(self._device)
            out = model.get_image_features(**inputs)
            # transformers versions differ: tensor or a pooling output object
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state[:, 0, :]
            out = out / out.norm(dim=-1, keepdim=True)
        return out[0].detach().cpu().numpy().astype(np.float32).flatten()
