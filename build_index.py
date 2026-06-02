"""
Build FAISS Index from Knowledge Base

One-time script to embed all agricultural disease entries and build
the FAISS index for retrieval.

Usage:
    python build_index.py
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import KNOWLEDGE_BASE_PATH, FAISS_INDEX_PATH
from models.embeddings import get_doc_embedder
from models.retrieval import FAISSRetriever


def make_document_text(entry: dict) -> str:
    """
    Concatenate all fields of a KB entry into a single document string
    for embedding.
    """
    parts = [
        f"Disease: {entry.get('disease', '')}",
        f"Crop: {entry.get('crop', '')}",
        f"Symptoms: {entry.get('symptoms', '')}",
        f"Cause: {entry.get('cause', '')}",
        f"Treatment: {entry.get('treatment', '')}",
        f"Prevention: {entry.get('prevention', '')}",
        f"Severity: {entry.get('severity', '')}",
    ]
    return " | ".join(parts)


def main():
    print("=" * 60)
    print("  🔨 Building FAISS Index from Knowledge Base")
    print("=" * 60)

    # 1. Load knowledge base
    print(f"\n[1/4] Loading knowledge base from {KNOWLEDGE_BASE_PATH} …")
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    print(f"       Found {len(kb)} entries.")

    # 2. Create document texts
    print("[2/4] Creating document representations …")
    documents = [make_document_text(entry) for entry in kb]
    for i, doc in enumerate(documents[:3]):
        print(f"       [{i}] {doc[:80]}…")

    # 3. Embed documents
    print("[3/4] Embedding documents (this may take a minute on CPU) …")
    embedder = get_doc_embedder()
    embeddings = embedder.embed_batch(documents)
    print(f"       Embeddings shape: {embeddings.shape}")

    # Also create CLIP text embeddings of just the disease+symptoms for
    # better matching with image queries
    from models.embeddings import get_text_embedder
    text_embedder = get_text_embedder()

    clip_texts = [
        f"{entry.get('disease', '')} on {entry.get('crop', '')}: {entry.get('symptoms', '')}"
        for entry in kb
    ]
    clip_embeddings = text_embedder.embed_batch(clip_texts)
    print(f"       CLIP text embeddings shape: {clip_embeddings.shape}")

    # Average the two embeddings for a richer representation
    combined = 0.5 * embeddings + 0.5 * clip_embeddings
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined = combined / norms
    combined = combined.astype(np.float32)
    print(f"       Combined embeddings shape: {combined.shape}")

    # 4. Build and save FAISS index
    print("[4/4] Building FAISS index …")
    retriever = FAISSRetriever()
    retriever.build_index(combined, kb)
    retriever.save(FAISS_INDEX_PATH)

    print(f"\n✅ Index saved to {FAISS_INDEX_PATH}")
    print("   You can now run: python app.py")


if __name__ == "__main__":
    main()
