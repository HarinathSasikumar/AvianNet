import numpy as np

def interpret_attention_scores(sequence: str, attention_scores: np.ndarray, top_n: int = 3):
    """
    Translates model attention logic back to biological contexts. 
    Highlights precisely which sub-regions forced the system trajectory.
    
    Args:
        sequence (str): The cleaned precursor DNA string.
        attention_scores (np.ndarray): Scaled 1D array of fractional importance probabilities.
        top_n (int): Number of most critical clusters to yield out.
    """
    print("\n--- MODEL EXPLAINER (ATTENTION MAP) ---")
    print("Prediction Interpretation Core:")
    
    # Process attention logic via argsort mapping indices dynamically backwards
    scores = np.asarray(attention_scores)
    sorted_indices = np.argsort(scores)[-top_n:][::-1]
    
    # Our timeline scaling ratio logic
    # K-mer Sequence (100) -> MaxPool (pool_size=2) -> roughly halves array size twice
    # 1 timestep returned functionally maps to roughly ~4 original raw base sequences
    SCALING_FACTOR = 4
    
    print(f"Top {top_n} Discovered Risky Sequences:")
    for rank, idx in enumerate(sorted_indices, start=1):
        importance_score = scores[idx]
        
        # DNA mapping window bounds
        real_start = idx * SCALING_FACTOR
        # Expand physical window slightly by +6 to absorb feature overlaps
        real_end = min(real_start + 6, len(sequence))
        
        dna_chunk = sequence[real_start:real_end]
        
        print(f"Position {real_start:02d}-{real_end:02d} \t-> {dna_chunk} \t(score: {importance_score:.4f})")
    
    print("---------------------------------------")
