"""
Layer 2 — Embedding Layer (Feature Extraction)

Converts multimodal inputs into unified vector representations:
  - ImageEmbedder:    PIL Image → 512-dim vector  (CLIP ViT-B/32)
  - TextEmbedder:     str       → 512-dim vector  (CLIP text encoder)
  - DocumentEmbedder: str       → 512-dim vector  (sentence-transformers → projection)

Note: transformers >= 5.x returns BaseModelOutputWithPooling from
get_text_features / get_image_features.  We extract .pooler_output.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLIP_MODEL_NAME,
    SENTENCE_MODEL_NAME,
    CLIP_EMBED_DIM,
    SENTENCE_EMBED_DIM,
    UNIFIED_EMBED_DIM,
    DEVICE,
)


def _extract_tensor(output):
    """
    Handle transformers 5.x API change: get_text_features / get_image_features
    may return BaseModelOutputWithPooling instead of a raw tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        # Fallback: mean-pool the last hidden state
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected output type from CLIP: {type(output)}")


class ImageEmbedder:
    """Encodes images using CLIP ViT-B/32 → 512-dim normalised vector."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        print(f"[ImageEmbedder] Loading CLIP model: {CLIP_MODEL_NAME} ...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()
        print("[ImageEmbedder] Ready.")

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        """Return L2-normalised 512-d embedding for a single PIL image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        raw = self.model.get_image_features(**inputs)
        features = _extract_tensor(raw)                            # (1, 512)
        features = features / features.norm(dim=-1, keepdim=True)  # L2-norm
        return features.cpu().numpy().squeeze(0)                   # (512,)

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Return embeddings for a list of PIL images -> (N, 512)."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        raw = self.model.get_image_features(**inputs)
        features = _extract_tensor(raw)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


class TextEmbedder:
    """Encodes short text queries using CLIP text encoder -> 512-dim."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        print(f"[TextEmbedder] Loading CLIP model: {CLIP_MODEL_NAME} ...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()
        print("[TextEmbedder] Ready.")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalised 512-d embedding for a single text string."""
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=77
        ).to(self.device)
        raw = self.model.get_text_features(**inputs)
        features = _extract_tensor(raw)                            # (1, 512)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze(0)                   # (512,)

    @torch.no_grad()
    def embed_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return embeddings for a list of text strings -> (N, 512)."""
        all_features = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=77
            ).to(self.device)
            raw = self.model.get_text_features(**inputs)
            features = _extract_tensor(raw)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)


class DocumentEmbedder:
    """
    Encodes longer documents / knowledge-base entries using
    sentence-transformers (all-MiniLM-L6-v2, 384-dim) with a learned
    linear projection to the unified 512-dim space.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        print(f"[DocumentEmbedder] Loading sentence model: {SENTENCE_MODEL_NAME} ...")
        self.model = SentenceTransformer(SENTENCE_MODEL_NAME, device=device)
        # linear projection 384 -> 512
        self.projection = nn.Linear(SENTENCE_EMBED_DIM, UNIFIED_EMBED_DIM).to(device)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection.eval()
        print("[DocumentEmbedder] Ready.")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalised 512-d embedding for a single document string."""
        raw = self.model.encode([text], convert_to_tensor=True).to(self.device)  # (1, 384)
        projected = self.projection(raw)                                          # (1, 512)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        return projected.cpu().numpy().squeeze(0)  # (512,)

    @torch.no_grad()
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for a list of document strings -> (N, 512)."""
        raw = self.model.encode(texts, convert_to_tensor=True).to(self.device)
        projected = self.projection(raw)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        return projected.cpu().numpy()


# -- Shared singleton loader -----------------------------------------------
_IMAGE_EMBEDDER = None
_TEXT_EMBEDDER = None
_DOC_EMBEDDER = None


def get_image_embedder() -> ImageEmbedder:
    global _IMAGE_EMBEDDER
    if _IMAGE_EMBEDDER is None:
        _IMAGE_EMBEDDER = ImageEmbedder()
    return _IMAGE_EMBEDDER


def get_text_embedder() -> TextEmbedder:
    global _TEXT_EMBEDDER
    if _TEXT_EMBEDDER is None:
        _TEXT_EMBEDDER = TextEmbedder()
    return _TEXT_EMBEDDER


def get_doc_embedder() -> DocumentEmbedder:
    global _DOC_EMBEDDER
    if _DOC_EMBEDDER is None:
        _DOC_EMBEDDER = DocumentEmbedder()
    return _DOC_EMBEDDER
