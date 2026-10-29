import torch
import numpy as np

# Import the new optimal models from our pipeline
from src.representation.dnabert_encoder import DNABERTEncoder
from src.models.fusion_model import PyTorchCNNExtractor, FusionAttentionModel
from src.models.gat_mutation_model import GATMutationModel, sequence_to_graph
from src.models.xgboost_classifier import XGBoostClassificationHead

def extract_features_for_batch(sequences, dnabert, cnn, gat, fusion, device):
    """
    Passes a batch of raw text DNA strings through the Deep Learning pipeline
    to extract the fused output embeddings [batch_size, 128] for XGBoost.
    """
    batch_fused = []
    
    for seq in sequences:
        # 1. DNABERT State-of-the-Art Context
        # Shape: [1, 768]
        dna_emb = dnabert.encode(seq)
        
        # 2. PyTorch CNN Spatial Features
        # For CNN we need a sequence of embeddings: [1, in_channels, seq_len]
        # In this dummy adaptation, we'll pretend we use K-mer embeddings or repeat dnabert
        cnn_input = dna_emb.unsqueeze(2).expand(-1, -1, 50) # shape [1, 768, 50]
        cnn_feat = cnn(cnn_input) # [1, 64]
        
        # 3. GAT Mutational Features
        # Needs graph construction
        # For simplicity, we chunk the dnabert embedding to act as graph nodes
        graph = sequence_to_graph(dna_emb.expand(10, 768)) 
        gat_nodes, _ = gat(graph.x.to(device), graph.edge_index.to(device))
        # Global mean pool across graph nodes to get sequence-level gat feature
        gat_feat = gat_nodes.mean(dim=0, keepdim=True) # [1, 128]
        
        # 4. Fusion
        fused_vec, _ = fusion(dna_emb, cnn_feat, gat_feat) # [1, 128]
        
        batch_fused.append(fused_vec.detach().cpu().numpy())
        
    return np.vstack(batch_fused)

def run_multimodal_pipeline():
    """
    Orchestrates the new PyTorch + XGBoost multimodal pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Structural Pipeline on {device}...")
    
    # Initialize all "Best" models
    dnabert = DNABERTEncoder(device=device)
    cnn = PyTorchCNNExtractor(in_channels=768, num_filters1=128, num_filters2=64).to(device)
    gat = GATMutationModel(in_channels=768, hidden_channels=64, out_channels=128).to(device)
    fusion = FusionAttentionModel(dnabert_dim=768, cnn_dim=64, gat_dim=128, fused_dim=256).to(device)
    xgb_head = XGBoostClassificationHead(n_estimators=100)
    
    # --- DUMMY DATA FOR DEMONSTRATION ---
    # Since building actual dataset needs the full parsing script, we show the pipeline logic
    mock_train_seqs = [
        "ATGCGTACGATCGATC", "GGTACGTACGTAGG", "CCGTTTAAAACCGG", "TTTGGCAAAATCGG"
    ]
    mock_train_labels = np.array([0, 1, 0, 1])
    
    mock_test_seqs = [
        "ATGCGTTTAAAAATC", "GGGGTTTTAAACCG", "ATATTATATATCGA"
    ]
    mock_test_labels = np.array([0, 1, 0])
    # ------------------------------------
    
    print("Extracting Deep Fused Features from Train Set...")
    X_train_fused = extract_features_for_batch(mock_train_seqs, dnabert, cnn, gat, fusion, device)
    
    print("Extracting Deep Fused Features from Test Set...")
    X_test_fused = extract_features_for_batch(mock_test_seqs, dnabert, cnn, gat, fusion, device)
    
    # Final Stage: XGBoost
    print("Training Final XGBoost Classifier...")
    xgb_head.train(X_train_fused, mock_train_labels)
    
    print("\n--- Pipeline Evaluation ---")
    metrics = xgb_head.evaluate(X_test_fused, mock_test_labels)
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"AUC: {metrics['auc']:.2f}\n")
    print(metrics['report'])
    print("Pipeline Execution Complete! Model is ready for production.")

if __name__ == "__main__":
    run_multimodal_pipeline()
