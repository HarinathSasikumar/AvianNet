"""
src/models/lightweight_fusion.py
=================================
Lightweight Concatenation Fusion Module.

REPLACES the FusionAttentionModel (4-head Transformer Cross-Attention).

Architecture:
  [DNABERT-2 (768d)] ─┐
  [CNN+BiLSTM  (64d)] ─┼─ Concatenate [960d] → ProjectionMLP → [128d]
  [GAT         (128d)]─┘

ProjectionMLP:
  Linear(960 → 512) → BatchNorm1d → ReLU → Dropout(0.3)
  Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.2)
  Linear(256 → 128) → (output: feature vector for XGBoost)

Design rationale:
  - Removes the 2-layer TransformerEncoder and associated positional
    attention overhead (~500k params).
  - Retains full information from all three branches via simple concat.
  - BatchNorm stabilises training without needing attention warm-up.
  - Output dim matches the old FusionAttentionModel (128) so XGBoost
    head is compatible with no changes required.
"""

import torch
import torch.nn as nn


class LightweightConcatFusion(nn.Module):
    """
    Lightweight feature-concatenation fusion module.

    Replaces the 4-head Transformer Cross-Attention fusion layer.
    Input dimensions must match the three upstream extractors:
        dnabert_dim : output dim of DNABERTEncoder.encode()  → 768
        cnn_dim     : output dim of PyTorchCNNExtractor       →  64
        gat_dim     : output dim of GATMutationModel (global) → 128
    Combined input to MLP: 768 + 64 + 128 = 960
    Output: 128-dimensional feature vector.
    """

    def __init__(
        self,
        dnabert_dim: int = 768,
        cnn_dim: int = 64,
        gat_dim: int = 128,
        hidden1: int = 512,
        hidden2: int = 256,
        out_dim: int = 128,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super(LightweightConcatFusion, self).__init__()

        concat_dim = dnabert_dim + cnn_dim + gat_dim  # 960 by default

        self.mlp = nn.Sequential(
            # --- Layer 1 ---
            nn.Linear(concat_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            # --- Layer 2 ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            # --- Output ---
            nn.Linear(hidden2, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        dnabert_feat: torch.Tensor,  # [batch, 768]
        cnn_feat: torch.Tensor,      # [batch,  64]
        gat_feat: torch.Tensor,      # [batch, 128]
    ) -> torch.Tensor:
        """
        Concatenate the three branch features and project to output_dim.

        Returns
        -------
        torch.Tensor  shape [batch, out_dim]
        """
        # Ensure all inputs are 2-D: [batch, dim]
        if dnabert_feat.dim() == 3:
            dnabert_feat = dnabert_feat.squeeze(1)
        if cnn_feat.dim() == 3:
            cnn_feat = cnn_feat.squeeze(1)
        if gat_feat.dim() == 3:
            gat_feat = gat_feat.squeeze(1)

        # Handle batch size = 1 edge case: switch BatchNorm to eval mode
        was_training = self.training
        if dnabert_feat.size(0) == 1 and was_training:
            self.eval()
            concatenated = torch.cat([dnabert_feat, cnn_feat, gat_feat], dim=1)
            output = self.mlp(concatenated)
            self.train()
        else:
            concatenated = torch.cat([dnabert_feat, cnn_feat, gat_feat], dim=1)
            output = self.mlp(concatenated)

        return output  # [batch, out_dim]

    def parameter_count(self) -> int:
        """Convenience method - total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== LightweightConcatFusion shape test ===")
    batch_size = 8
    mock_dnabert = torch.rand(batch_size, 768)
    mock_cnn     = torch.rand(batch_size, 64)
    mock_gat     = torch.rand(batch_size, 128)

    fusion = LightweightConcatFusion()
    fusion.train()
    out = fusion(mock_dnabert, mock_cnn, mock_gat)

    assert out.shape == (batch_size, 128), f"Shape mismatch: {out.shape}"
    print(f"Output shape  : {out.shape}   [PASS]")
    print(f"Param count   : {fusion.parameter_count():,}")
    print("Test PASSED.")
