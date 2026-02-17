"""CLIP-based embeddings for text and images in a unified vector space."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    """Unified CLIP embedder for text and images."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self._model = CLIPModel.from_pretrained(model_id)
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model.eval()

    def embed_image(
        self, image: Union[str, Path, Image.Image]
    ) -> np.ndarray:
        """Embed a single image; input can be path or PIL Image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be path (str/Path) or PIL.Image")

        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def embed_text(self, text: str, max_length: int = 77) -> np.ndarray:
        """Embed a single text string."""
        inputs = self._processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def embed_texts(self, texts: list[str], max_length: int = 77) -> np.ndarray:
        """Embed a list of texts (batch)."""
        if not texts:
            return np.array([]).reshape(0, self._model.config.projection_dim)
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.numpy()

    @property
    def embedding_dim(self) -> int:
        return self._model.config.projection_dim
