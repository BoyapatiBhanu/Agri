"""
Layer 4 — Cross-Modal Transformer Layer

Enables deep interaction between modalities via multi-head self-attention.
Image patch features and text token features attend to each other, allowing
the model to discover correspondences (e.g. leaf spot region ↔ "brown lesion").

Architecture
------------
- Concatenate image and text token sequences
- Pass through a 2-layer Transformer encoder
- Output attended features per token
- Pool into a single 512-d representation
"""

import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    UNIFIED_EMBED_DIM,
    CROSS_MODAL_HEADS,
    CROSS_MODAL_LAYERS,
    CROSS_MODAL_DIM_FF,
    CROSS_MODAL_DROPOUT,
    DEVICE,
)


class CrossModalTransformer(nn.Module):
    """
    Cross-modal transformer that fuses token-level representations
    from different modalities via multi-head self-attention.

    Input : (batch, seq_len, d_model)   — concatenated modality tokens
    Output: (batch, d_model)            — pooled cross-attended representation
    """

    def __init__(
        self,
        d_model: int = UNIFIED_EMBED_DIM,
        nhead: int = CROSS_MODAL_HEADS,
        num_layers: int = CROSS_MODAL_LAYERS,
        dim_feedforward: int = CROSS_MODAL_DIM_FF,
        dropout: float = CROSS_MODAL_DROPOUT,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (batch, seq_len, d_model)

        Returns
        -------
        pooled : (batch, d_model)
        """
        batch_size = tokens.size(0)
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, tokens], dim=1)              # (B, 1+S, D)
        x = self.transformer(x)                          # (B, 1+S, D)
        pooled = self.layer_norm(x[:, 0, :])             # CLS output → (B, D)
        return pooled


class CrossModalProcessor:
    """
    High-level wrapper: takes numpy embeddings from different modalities,
    converts them into token sequences, and runs the cross-modal transformer.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.model = CrossModalTransformer().to(device)
        self.model.eval()

    @torch.no_grad()
    def process(
        self,
        text_emb: np.ndarray | None = None,
        image_emb: np.ndarray | None = None,
        doc_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Accepts individual 512-d embeddings, forms a token sequence,
        and returns the cross-attended 512-d output.
        """
        tokens = []
        if text_emb is not None:
            tokens.append(torch.tensor(text_emb, dtype=torch.float32))
        if image_emb is not None:
            tokens.append(torch.tensor(image_emb, dtype=torch.float32))
        if doc_emb is not None:
            tokens.append(torch.tensor(doc_emb, dtype=torch.float32))

        if len(tokens) == 0:
            return np.zeros(UNIFIED_EMBED_DIM, dtype=np.float32)

        # Stack into (1, num_modalities, 512)
        token_seq = torch.stack(tokens, dim=0).unsqueeze(0).to(self.device)
        output = self.model(token_seq)  # (1, 512)
        output = output / output.norm(dim=-1, keepdim=True)
        return output.cpu().numpy().squeeze(0)
