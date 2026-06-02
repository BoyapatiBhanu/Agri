"""
Layer 5 — Multi-Vector Representation Layer  ⭐ (Key Innovation)

Instead of compressing the query to a single vector, we produce *k*
sub-vectors that capture different semantic aspects:

    M = {z₁, z₂, …, z_k}

Scoring uses **late interaction** (ColBERT-style):

    score(Q, D) = Σᵢ max_j  zᵢ^Q · zⱼ^D

This preserves fine-grained detail — e.g. matching a specific leaf
spot pattern to a specific disease mention — while remaining efficient
for FAISS first-stage retrieval.
"""

import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import UNIFIED_EMBED_DIM, NUM_VECTORS, MULTI_VEC_DIM, DEVICE


class MultiVectorProjector(nn.Module):
    """
    Projects a single 512-d representation into k sub-vectors.

    Input  : (batch, 512)
    Output : (batch, k, sub_dim)
    """

    def __init__(
        self,
        input_dim: int = UNIFIED_EMBED_DIM,
        num_vectors: int = NUM_VECTORS,
        sub_dim: int = MULTI_VEC_DIM,
    ):
        super().__init__()
        self.num_vectors = num_vectors
        self.sub_dim = sub_dim

        # One linear head per sub-vector
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, sub_dim),
                nn.GELU(),
                nn.Linear(sub_dim, sub_dim),
            )
            for _ in range(num_vectors)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        multi_vec : (batch, num_vectors, sub_dim)  — L2-normalized per sub-vec
        """
        vecs = [head(x) for head in self.heads]                # list of (B, sub_dim)
        multi_vec = torch.stack(vecs, dim=1)                   # (B, k, sub_dim)
        multi_vec = multi_vec / (multi_vec.norm(dim=-1, keepdim=True) + 1e-8)
        return multi_vec


class MultiVectorRepresentation:
    """
    High-level interface for producing and scoring multi-vector
    representations.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.projector = MultiVectorProjector().to(device)
        self.projector.eval()

    @torch.no_grad()
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        Convert a single 512-d embedding into k sub-vectors.

        Returns
        -------
        np.ndarray of shape (k, sub_dim)
        """
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        multi_vec = self.projector(x)  # (1, k, sub_dim)
        return multi_vec.cpu().numpy().squeeze(0)  # (k, sub_dim)

    @staticmethod
    def late_interaction_score(
        query_vecs: np.ndarray,
        doc_vecs: np.ndarray,
    ) -> float:
        """
        ColBERT-style late-interaction scoring.

        score = Σ_i  max_j  (q_i · d_j)

        Parameters
        ----------
        query_vecs : (k_q, dim)
        doc_vecs   : (k_d, dim)

        Returns
        -------
        float  — similarity score
        """
        # (k_q, k_d)
        sim_matrix = query_vecs @ doc_vecs.T
        # For each query vector, take max similarity across doc vectors
        max_sims = sim_matrix.max(axis=1)
        return float(max_sims.sum())

    @staticmethod
    def collapse_to_single(multi_vec: np.ndarray) -> np.ndarray:
        """
        Collapse multi-vector back to a single vector for FAISS indexing
        by averaging and normalizing.

        Parameters
        ----------
        multi_vec : (k, sub_dim)

        Returns
        -------
        np.ndarray of shape (sub_dim,)
        """
        collapsed = multi_vec.mean(axis=0)
        norm = np.linalg.norm(collapsed)
        if norm > 0:
            collapsed = collapsed / norm
        return collapsed
