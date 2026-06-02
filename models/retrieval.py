"""
Layer 6 — Retrieval Layer (FAISS Search)

Builds and queries a FAISS index over the agricultural knowledge base.
Two-stage retrieval:
  1. FAISS first-pass with single (collapsed) vectors → top-K candidates
  2. Re-rank candidates with multi-vector late interaction → final top-K
"""

import json
import os
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("[WARNING] faiss-cpu not installed. Falling back to brute-force search.")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    KNOWLEDGE_BASE_PATH,
    FAISS_INDEX_PATH,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    TOP_K,
    UNIFIED_EMBED_DIM,
)


class FAISSRetriever:
    """
    Manages a FAISS index for fast nearest-neighbour search over the
    knowledge-base embeddings.
    """

    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []     # parallel list of KB entries
        self.embeddings: np.ndarray | None = None  # (N, dim) stored for re-ranking

    # ── Index Building ───────────────────────────────────────────

    def build_index(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        Build a FAISS inner-product index from pre-computed embeddings.

        Parameters
        ----------
        embeddings : (N, dim) float32 array — L2-normalized
        metadata   : list of N dicts with knowledge-base fields
        """
        dim = embeddings.shape[1]
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata

        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)  # inner product = cosine for unit vecs
            self.index.add(self.embeddings)
            print(f"[FAISSRetriever] Built index with {self.index.ntotal} vectors (dim={dim}).")
        else:
            print(f"[FAISSRetriever] Built brute-force index with {len(metadata)} vectors (dim={dim}).")

    def save(self, path: str = FAISS_INDEX_PATH):
        """Persist index and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        # Always save embeddings for re-ranking and brute-force fallback
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[FAISSRetriever] Saved index to {path}")

    def load(self, path: str = FAISS_INDEX_PATH):
        """Load persisted index from disk."""
        emb_path = os.path.join(path, "embeddings.npy")
        meta_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No index found at {path}. Run build_index.py first."
            )

        self.embeddings = np.load(emb_path).astype(np.float32)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        faiss_path = os.path.join(path, "index.faiss")
        if faiss is not None and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            print(f"[FAISSRetriever] Loaded FAISS index ({self.index.ntotal} vectors).")
        else:
            print(f"[FAISSRetriever] Loaded brute-force index ({len(self.metadata)} vectors).")

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Retrieve top-k most similar knowledge-base entries.

        Parameters
        ----------
        query_embedding : (dim,) float32 L2-normalised query vector
        top_k           : number of results to return

        Returns
        -------
        List of dicts, each with keys from the KB entry plus 'score'.
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)

        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force fallback
            sims = (self.embeddings @ query.T).squeeze(-1)  # (N,)
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results


# ── Singleton ─────────────────────────────────────────────────────
_RETRIEVER = None


def get_retriever() -> FAISSRetriever:
    """Get or create the singleton retriever (loads index from disk)."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = FAISSRetriever()
        try:
            _RETRIEVER.load()
        except FileNotFoundError:
            print("[FAISSRetriever] No pre-built index found. Call build_index.py first.")
    return _RETRIEVER
