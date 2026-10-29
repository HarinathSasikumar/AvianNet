import re

def clean_dna(sequence: str) -> str:
    """
    Cleans a DNA sequence by converting it to uppercase and removing any
    characters that are not standard DNA bases (A, T, G, C).
    
    Args:
        sequence (str): The input DNA sequence.
        
    Returns:
        str: A clean, uppercase DNA sequence containing only A, T, G, C.
             Returns an empty string if the input is empty or invalid.
    """
    if not sequence or not isinstance(sequence, str):
        return ""
    
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Remove any character that is not A, T, G, or C
    clean_seq = re.sub(r'[^ATGC]', '', sequence)
    
    return clean_seq

def validate_dna(sequence: str) -> bool:
    """
    Validates whether a given sequence consists entirely of valid DNA bases (A, T, G, C).
    
    Args:
        sequence (str): The input sequence to validate.
        
    Returns:
        bool: True if the sequence is valid and not empty, False otherwise.
    """
    if not sequence or not isinstance(sequence, str):
        return False
        
    sequence = sequence.upper()
    
    # Check if there are any non-ATGC characters
    if re.search(r'[^ATGC]', sequence):
        return False
        
    return True

def pad_or_trim(sequence: str, max_length: int, pad_char: str = 'N') -> str:
    """
    Ensures a DNA sequence is of a fixed length by trimming it if it's too long
    or padding it if it's too short.
    
    Args:
        sequence (str): The input DNA sequence.
        max_length (int): The target length for the sequence.
        pad_char (str, optional): The character to use for padding. Defaults to 'N'.
        
    Returns:
        str: The trimmed or padded sequence of exactly max_length characters.
    """
    if not isinstance(sequence, str):
        sequence = ""
        
    if max_length <= 0:
        return ""
        
    if len(sequence) > max_length:
        # Trim sequence
        return sequence[:max_length]
    elif len(sequence) < max_length:
        # Pad sequence
        padding = pad_char * (max_length - len(sequence))
        return sequence + padding
        
    return sequence

if __name__ == "__main__":
    # Small test example
    print("--- DNA Preprocessing Module Tests ---")
    
    test_sequence_1 = "atgc NNY TGa-C"
    test_sequence_2 = "ATGCGTA"
    empty_sequence = ""
    invalid_input = None
    
    # Test clean_dna
    print(f"\n1. Testing clean_dna:")
    print(f"Original: '{test_sequence_1}' -> Cleaned: '{clean_dna(test_sequence_1)}'")
    
    # Test validate_dna
    print(f"\n2. Testing validate_dna:")
    print(f"'{test_sequence_2}' is valid? {validate_dna(test_sequence_2)}")
    print(f"'{test_sequence_1}' is valid? {validate_dna(test_sequence_1)}")
    
    # Test pad_or_trim
    print(f"\n3. Testing pad_or_trim (target length: 10):")
    print(f"Sequence: '{test_sequence_2}', Length: {len(test_sequence_2)}")
    padded = pad_or_trim(test_sequence_2, max_length=10)
    print(f"Padded: '{padded}', New Length: {len(padded)}")
    
    print(f"\nTesting pad_or_trim (target length: 5):")
    print(f"Sequence: '{test_sequence_2}', Length: {len(test_sequence_2)}")
    trimmed = pad_or_trim(test_sequence_2, max_length=5)
    print(f"Trimmed: '{trimmed}', New Length: {len(trimmed)}")
    
    # Test edge cases
    print(f"\n4. Edge Cases:")
    print(f"Empty sequence clean: '{clean_dna(empty_sequence)}'")
    print(f"Invalid input clean: '{clean_dna(invalid_input)}'")
    print(f"Empty sequence valid: {validate_dna(empty_sequence)}")
