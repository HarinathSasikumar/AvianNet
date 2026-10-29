import itertools
from typing import List, Dict

def generate_kmers(sequence: str, k: int) -> List[str]:
    """
    Generates all k-mers of a given length from a DNA sequence.
    Extracts overlapping substrings of length k with a stride of 1.
    
    Args:
        sequence (str): The input DNA sequence.
        k (int): The length of each k-mer.
        
    Returns:
        List[str]: A list of generated k-mers. Returns an empty list if 
                   the sequence is shorter than k or invalid.
    """
    if not sequence or not isinstance(sequence, str) or k <= 0 or len(sequence) < k:
        return []
        
    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return kmers

def build_vocab(k: int) -> Dict[str, int]:
    """
    Generates a vocabulary of all possible k-mer configurations from standard
    DNA bases (A, C, G, T) and maps each to a unique integer index. 
    
    Args:
        k (int): The length of the k-mers to build vocabulary for.
        
    Returns:
        Dict[str, int]: A consistent mapping from k-mer strings to integer indices.
                        Keys are alphabetically sorted to maintain dictionary structure.
    """
    if k <= 0:
        return {}
        
    bases = ['A', 'C', 'G', 'T']
    
    # Use itertools.product to generate all possible combinations with repetition
    # product('ACGT', repeat=2) yields AA, AC, AG, AT, CA, CC...
    combinations = itertools.product(bases, repeat=k)
    
    # Join tuples into strings and ensure they are sorted alphabetically
    all_kmers = sorted([''.join(c) for c in combinations])
    
    # Build vocabulary: { 'AA': 1, 'AC': 2, ... }
    # Using 1-based indexing, leaving 0 for padding or unknowns if needed
    vocab = {kmer: idx + 1 for idx, kmer in enumerate(all_kmers)}
    
    return vocab

def encode_kmers(kmers: List[str], vocab: Dict[str, int], unknown_index: int = 0) -> List[int]:
    """
    Converts a list of k-mer strings into integer representations using a vocabulary.
    
    Args:
        kmers (List[str]): List of k-mers to encode.
        vocab (Dict[str, int]): The sequence vocabulary matrix mapping k-mers to indices.
        unknown_index (int): Output value if a k-mer is not found in the vocabulary.
        
    Returns:
        List[int]: A list of integer indices representing the sequence.
    """
    if not kmers or not vocab:
        return []
        
    # Use the lookup dict to get indices. Default missing to unknown_index.
    return [vocab.get(kmer, unknown_index) for kmer in kmers]

def pad_encoded_sequence(sequence: List[int], max_length: int) -> List[int]:
    """
    Pads or truncates an encoded numerical sequence to a fixed length.
    Padding is done with 0s appended to the end of the sequence.
    
    Args:
        sequence (List[int]): The encoded numerical sequence.
        max_length (int): The target length for the sequence.
        
    Returns:
        List[int]: The modified sequence of exactly max_length elements.
    """
    if not sequence:
        return [0] * max_length
        
    if len(sequence) > max_length:
        return sequence[:max_length]
    elif len(sequence) < max_length:
        padding = [0] * (max_length - len(sequence))
        return sequence + padding
        
    return sequence

if __name__ == "__main__":
    # Test Example
    print("--- K-mer Representation Tests ---")
    
    test_sequence = "ATGCGTA"
    k_length = 3
    
    # 1. Test Vocabulary Building
    print(f"\n1. Building Vocab for k={k_length}:")
    vocab_matrix = build_vocab(k_length)
    print(f"Total possible combinations: {len(vocab_matrix)}")
    
    # Sample the first 5 and last 5 elements of vocab to prove it runs correctly
    vocab_items = list(vocab_matrix.items())
    print(f"First 5 mappings:    {vocab_items[:5]}")
    print(f"Last 5 mappings:     {vocab_items[-5:]}")
    
    # 2. Test K-mer Generation
    print(f"\n2. Extracting k-mers:")
    print(f"Sequence: '{test_sequence}', Target K={k_length}")
    extracted_kmers = generate_kmers(test_sequence, k_length)
    print(f"Generated K-mers: {extracted_kmers}")
    
    # 3. Test Encoding logic
    print(f"\n3. Encoding Sequence:")
    encoded_sequence = encode_kmers(extracted_kmers, vocab_matrix)
    print(f"Encoded integers: {encoded_sequence}")
    
    # Handling Unmapped Chars ('N')
    corrupted_sequence = "ATNNGTA"
    print(f"\n4. Edge case sequence containing 'N' ('{corrupted_sequence}'):")
    corrupt_kmers = generate_kmers(corrupted_sequence, k_length)
    encoded_corrupted = encode_kmers(corrupt_kmers, vocab_matrix)
    print(f"Extracted: {corrupt_kmers}")
    print(f"Encoded (0 implies UNK): {encoded_corrupted}")
