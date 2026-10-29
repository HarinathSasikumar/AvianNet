import torch
import torch.nn as nn

class FusionAttentionModel(nn.Module):
    """
    Multimodal Fusion Attention Network.
    Fuses representations from:
      1. DNABERT Context Embeddings (Global sequence semantics)
      2. 1D CNN Maps (Local sequence motifs)
      3. GAT Mutational Nodes (Structural relationships)
    """
    def __init__(self, dnabert_dim=768, cnn_dim=128, gat_dim=128, fused_dim=256, num_heads=4):
        super(FusionAttentionModel, self).__init__()
        
        # Projection layers to align dimensions into a common fused_dim
        self.proj_dnabert = nn.Linear(dnabert_dim, fused_dim)
        self.proj_cnn = nn.Linear(cnn_dim, fused_dim)
        self.proj_gat = nn.Linear(gat_dim, fused_dim)
        
        # Transformer Encoder layer acting as a Multi-Head Cross-Attention Fusion layer
        # The 3 projected vectors will be treated as sequence tokens [batch_size, 3, fused_dim]
        # and self-attended so they can mix information.
        encoder_layer = nn.TransformerEncoderLayer(d_model=fused_dim, nhead=num_heads, batch_first=True)
        self.fusion_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output compression to a dense feature vector for XGBoost classifier
        self.output_layer = nn.Sequential(
            nn.Linear(fused_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)  # Final feature vector size
        )
        
    def forward(self, dnabert_feat, cnn_feat, gat_feat):
        """
        dnabert_feat: [batch_size, dnabert_dim]
        cnn_feat: [batch_size, cnn_dim]    (pooled CNN features)
        gat_feat: [batch_size, gat_dim]    (pooled GAT features)
        """
        # Project
        h_dna = self.proj_dnabert(dnabert_feat).unsqueeze(1)    # [batch, 1, fused_dim]
        h_cnn = self.proj_cnn(cnn_feat).unsqueeze(1)            # [batch, 1, fused_dim]
        h_gat = self.proj_gat(gat_feat).unsqueeze(1)            # [batch, 1, fused_dim]
        
        # Concat tokens
        stacked_features = torch.cat([h_dna, h_cnn, h_gat], dim=1)  # [batch, 3, fused_dim]
        
        # Fuse via Self-Attention (Allows DNABERT context mapped to CNN motifs, etc.)
        fused_tokens = self.fusion_attention(stacked_features)      # [batch, 3, fused_dim]
        
        # Flatten and compress
        batch_size = fused_tokens.size(0)
        flattened = fused_tokens.view(batch_size, -1)               # [batch, 3 * fused_dim]
        
        # Output dense vector ready for XGBoost pipeline
        dense_vector = self.output_layer(flattened)                 # [batch, 128]
        
        return dense_vector, fused_tokens


class PyTorchCNNExtractor(nn.Module):
    """
    Ported 1D CNN feature extractor to align with the PyTorch pipeline.
    Identical spatial motif gathering logic.
    """
    def __init__(self, in_channels=64, num_filters1=128, num_filters2=64):
        super(PyTorchCNNExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(num_filters1, num_filters2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Global max pooling to get a fixed size vector regardless of sequence length
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x expected shape: [batch, in_channels, seq_len]
        # (needs transpose if coming straight from embedding [batch, seq_len, in_channels])
        
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.global_pool(x).squeeze(-1) # [batch, out_channels]
        return x

if __name__ == "__main__":
    bs = 8
    mock_dnabert = torch.rand((bs, 768))
    mock_cnn = torch.rand((bs, 64))
    mock_gat = torch.rand((bs, 128))
    
    fusion = FusionAttentionModel(cnn_dim=64)
    final_vec, attention_states = fusion(mock_dnabert, mock_cnn, mock_gat)
    print(f"Final Fused Vector for XGBoost: {final_vec.shape}") # Expected: [8, 128]
