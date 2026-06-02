"""
Layer 3 — Multimodal Fusion Layer

Combines text, image, and document embeddings into a single fused
representation using weighted linear combination:

    E_fusion = α·E_T + β·E_I + γ·E_D

Gracefully handles missing modalities by re-normalizing weights.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FUSION_ALPHA, FUSION_BETA, FUSION_GAMMA, UNIFIED_EMBED_DIM


class MultimodalFusion:
    """
    Weighted fusion of multimodal embeddings.

    Parameters
    ----------
    alpha : float  – weight for text embedding
    beta  : float  – weight for image embedding
    gamma : float  – weight for document embedding
    """

    def __init__(
        self,
        alpha: float = FUSION_ALPHA,
        beta: float = FUSION_BETA,
        gamma: float = FUSION_GAMMA,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fuse(
        self,
        text_emb: np.ndarray | None = None,
        image_emb: np.ndarray | None = None,
        doc_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fuse available embeddings.

        Missing modalities are ignored and weights re-normalized so that
        the result is always a unit-norm 512-d vector.

        Returns
        -------
        np.ndarray of shape (512,)
        """
        components: list[tuple[float, np.ndarray]] = []

        if text_emb is not None:
            components.append((self.alpha, text_emb))
        if image_emb is not None:
            components.append((self.beta, image_emb))
        if doc_emb is not None:
            components.append((self.gamma, doc_emb))

        if len(components) == 0:
            # Return zero vector if nothing is available
            return np.zeros(UNIFIED_EMBED_DIM, dtype=np.float32)

        # Re-normalize weights
        total_weight = sum(w for w, _ in components)
        fused = sum((w / total_weight) * emb for w, emb in components)

        # L2-normalize the result
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused.astype(np.float32)

    def __repr__(self):
        return (
            f"MultimodalFusion(a={self.alpha:.2f}, b={self.beta:.2f}, g={self.gamma:.2f})"
        )
