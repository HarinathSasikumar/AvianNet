import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np

def build_cnn_feature_extractor(vocab_size: int, sequence_length: int) -> Model:
    """
    Builds a 1D Convolutional Neural Network model designed to extract 
    hierarchical features from encoded DNA sequences (e.g., K-mers).
    
    The network generates a high-dimensional feature vector rather than 
    making a direct classification, allowing it to be used flexibly in 
    different downstream tasks.
    
    Args:
        vocab_size (int): Size of the k-mer vocabulary (usually max index + 1).
        sequence_length (int): Fixed length of the padded input sequences.
        
    Returns:
        tf.keras.Model: The CNN feature extractor model.
    """
    # Define Sequence Input
    inputs = Input(shape=(sequence_length,), name="sequence_input")
    
    # 1. Embedding Layer
    # Transform integer indices into dense vectors of dimension 64
    x = Embedding(input_dim=vocab_size, output_dim=64, name="kmer_embedding")(inputs)
    
    # 2. First Convolutional Block
    # Extracts local spatial patterns (motifs) from the sequence
    x = Conv1D(filters=128, kernel_size=3, activation='relu', name="conv1d_1")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    
    # 3. Second Convolutional Block 
    # Learns more complex composite motifs and reduces dimensionality further
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_2")(x)
    
    # 4. Flatten and Dense Layer
    # Convert temporal feature maps into a single 1D vector representation
    x = Flatten(name="flatten_features")(x)
    features = Dense(units=128, activation='relu', name="feature_vector")(x)
    
    # Assemble computational graph into a cohesive model
    model = Model(inputs=inputs, outputs=features, name="1d_cnn_dna_extractor")
    return model


if __name__ == "__main__":
    # Test execution logic
    print("--- 1D CNN Feature Extractor Initializing ---")
    
    # Simulated constants (e.g., 64 k-mers of base length + 1 padding token)
    TEST_VOCAB_SIZE = 65 
    TEST_SEQUENCE_LENGTH = 100
    
    # 1. Build the model
    cnn_model = build_cnn_feature_extractor(vocab_size=TEST_VOCAB_SIZE, 
                                            sequence_length=TEST_SEQUENCE_LENGTH)
    
    # 2. Print architecture summary
    print("\n[Architecture Details]")
    cnn_model.summary()
    
    # 3. Test forward pass robustness using dummy tensors
    print("\n[Executing Forward Pass Test]")
    
    # Generate 4 random dummy sequences structurally matching our constraints
    dummy_input_batch = np.random.randint(0, TEST_VOCAB_SIZE, size=(4, TEST_SEQUENCE_LENGTH))
    print(f"Input batch shape via dataset: {dummy_input_batch.shape} (batch_size, sequence_length)")
    
    # Inject sequences through the untrailed model graph
    extracted_features = cnn_model(dummy_input_batch)
    
    print(f"Output features shape resolved as: {extracted_features.shape}")
    print("Logic validation successful. Matrix graph properly compiled and traversable.")
