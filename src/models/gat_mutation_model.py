import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class GATMutationModel(nn.Module):
    """
    Graph Attention Network (GAT) for Mutation Identification.
    Treats the DNA sequence (broken into k-mers/chunks) as a graph where 
    adjacent sequence parts are connected. Evaluates nodes (k-mers) structurally.
    """
    def __init__(self, in_channels=64, hidden_channels=128, heads=4, out_channels=128):
        super(GATMutationModel, self).__init__()
        # Input: features of each node (e.g., token embedding size)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        # Output layer concatenates heads. To bring back to out_channels:
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.3)
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity matrix [2, num_edges]
        Returns: [num_nodes, out_channels] aggregated mutational node features,
                 and the attention weights!
        """
        x, alpha1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.activation(x)
        x, alpha2 = self.gat2(x, edge_index, return_attention_weights=True)
        return x, alpha2

def sequence_to_graph(sequence_embeddings):
    """
    Utility to format a linear sequence of continuous embeddings into a basic Graph
    for GAT evaluation. Nodes = sequence positions. Edges = sequential connections.
    sequence_embeddings: Tensor of shape (seq_len, embed_dim)
    Returns: torch_geometric.data.Data object
    """
    seq_len = sequence_embeddings.size(0)
    
    # Create sequential edges (node_i connected to node_i+1 and vice versa)
    sources = []
    targets = []
    
    for i in range(seq_len - 1):
        sources.extend([i, i+1])
        targets.extend([i+1, i])
        
    # Also add self-loops
    sources.extend(list(range(seq_len)))
    targets.extend(list(range(seq_len)))
    
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    data = Data(x=sequence_embeddings, edge_index=edge_index)
    return data

if __name__ == "__main__":
    # Test GAT Model
    seq_len = 50
    embed_dim = 64
    mock_embeddings = torch.rand((seq_len, embed_dim))
    
    graph_data = sequence_to_graph(mock_embeddings)
    model = GATMutationModel(in_channels=embed_dim, hidden_channels=32, heads=2, out_channels=128)
    
    out, attention = model(graph_data.x, graph_data.edge_index)
    print(f"GAT Nodes Output Shape: {out.shape}")  # Should be [seq_len, 128]
    print(f"GAT Attention Weights Shape: {attention[0].shape}")  # Edge alignments
