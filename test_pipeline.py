import numpy as np
from src.preprocessing.dna_preprocessing import clean_dna, validate_dna
from src.representation.kmer_representation import build_vocab, generate_kmers, encode_kmers, pad_encoded_sequence
from src.models.final_model import build_final_model
from src.utils.attention_utils import interpret_attention_scores

def interpret_prediction(probability: float):
    # Determine absolute risk
    if probability < 0.3:
        return "Healthy"
    elif probability < 0.7:
        return "Risk"
    else:
        return "Infected"

def run_integration_pipeline():
    print("=== FINAL DNA CLASSIFICATION PIPELINE ===\n")
    
    # Base configuration and logic hook logic
    sample_dna = "ATGCGTATGACCGTAGTGACTAGCTAGCTGATCGATCGTAGCTAGCTGCATCGTAGCTGACTAGCTAGCTACGATCGTAGCTAGTCAGTCA"
    print(f"Loading Biological Sample...")
    print(f"Sequence Length: {len(sample_dna)} base pairs\n")
    
    # Structure mapping & normalization
    cleaned_dna = clean_dna(sample_dna)
    if not validate_dna(cleaned_dna):
        print("Sequence is corrupt. Aborting pipeline.")
        return
        
    # Standard representation encoding arrays
    k = 3
    extracted_kmers = generate_kmers(cleaned_dna, k)
    vocab = build_vocab(k)
    encoded_array = encode_kmers(extracted_kmers, vocab)
    
    # Format batch processing constants structurally
    max_length = 100
    padded_array = pad_encoded_sequence(encoded_array, max_length)
    cnn_input = np.array(padded_array).reshape(1, max_length)
    
    # Initialize Core System
    print("Initializing Core Intelligence System...")
    vocab_size = len(vocab) + 1
    model = build_final_model(vocab_size=vocab_size, sequence_length=max_length)
    
    # Predict output tensors natively
    predictions, attention = model.predict(cnn_input, verbose=0)
    
    # Map probability scalar metrics dynamically
    probability = float(predictions[0][0])
    label = interpret_prediction(probability)
    
    # Presentation Phase Output Data
    print("\n--- PIPELINE DIAGNOSTIC RESULTS ---")
    print(f"Prediction: {label.upper()}")
    print(f"Risk Probability: {probability:.4f}")
    
    attention_arr = attention[0]
    interpret_attention_scores(sequence=cleaned_dna, attention_scores=attention_arr, top_n=3)

if __name__ == "__main__":
    run_integration_pipeline()
