"""
Configuration for the Multimodal Agricultural RAG Pipeline.
All hyperparameters, model names, and paths are centralized here.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
FAISS_META_FILE = os.path.join(FAISS_INDEX_PATH, "metadata.pkl")

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model Names (Hugging Face)
# ──────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# Embedding Dimensions
# ──────────────────────────────────────────────
CLIP_EMBED_DIM = 512          # CLIP output dimension
SENTENCE_EMBED_DIM = 384      # MiniLM output dimension
UNIFIED_EMBED_DIM = 512       # Unified projection dimension

# ──────────────────────────────────────────────
# Fusion Weights  (Layer 3)
#   E_fusion = α·E_T + β·E_I + γ·E_D
# ──────────────────────────────────────────────
FUSION_ALPHA = 0.40   # text weight
FUSION_BETA = 0.40    # image weight
FUSION_GAMMA = 0.20   # document weight

# ──────────────────────────────────────────────
# Cross-Modal Transformer (Layer 4)
# ──────────────────────────────────────────────
CROSS_MODAL_HEADS = 8
CROSS_MODAL_LAYERS = 2
CROSS_MODAL_DIM_FF = 2048
CROSS_MODAL_DROPOUT = 0.1

# ──────────────────────────────────────────────
# Multi-Vector Representation (Layer 5)
# ──────────────────────────────────────────────
NUM_VECTORS = 8               # k vectors per query
MULTI_VEC_DIM = 128           # dimension of each sub-vector

# ──────────────────────────────────────────────
# FAISS Retrieval (Layer 6)
# ──────────────────────────────────────────────
TOP_K = 5                     # number of evidence docs to retrieve

# ──────────────────────────────────────────────
# RAG / Inference (Layers 7-8)
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5    # minimum confidence to report diagnosis
MAX_EVIDENCE_TOKENS = 1024    # max tokens from retrieved evidence
