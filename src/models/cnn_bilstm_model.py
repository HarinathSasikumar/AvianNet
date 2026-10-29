import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense
import numpy as np

def build_cnn_bilstm_feature_extractor(vocab_size: int, sequence_length: int) -> Model:
    """
    Builds a CNN + BiLSTM model for advanced DNA sequence classification.
     Extracts complex regional dependencies utilizing Bidirectional LSTMs
    rather than flattening the CNN temporal features blindly.
    
    Args:
        vocab_size (int): Size of the k-mer vocabulary (+1 for padding constraints).
        sequence_length (int): Fixed length of sequential padding.
        
    Returns:
        tf.keras.Model: The CNN+BiLSTM model natively tracking batch dimensions.
    """
    # 1. Base Input Vector Sequence
    inputs = Input(shape=(sequence_length,), name="sequence_input")
    
    # 2. Embedding Layer
    x = Embedding(input_dim=vocab_size, output_dim=64, name="kmer_embedding")(inputs)
    
    # 3. First CNN Block
    x = Conv1D(filters=128, kernel_size=3, activation='relu', name="conv1d_1")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    
    # 4. Second CNN Block
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_2")(x)
    
    # 5. Temporal Understanding / Sequence Intelligence Block
    # Output of Conv blocks is inherently 3D (batch_size, timesteps, filters).
    # Return sequences=False enforces temporal reduction to a purely 2D mapping (batch_size, units*2).
    x = Bidirectional(LSTM(units=64, return_sequences=False), name="bilstm_block")(x)
    
    # 6. Dense Feature Generation Vector
    features = Dense(units=128, activation='relu', name="feature_vector")(x)
    
    # Compile Topological Graph Pipeline
    model = Model(inputs=inputs, outputs=features, name="cnn_bilstm_dna_extractor")
    return model

if __name__ == "__main__":
    import numpy as np

    print("=== Testing CNN + BiLSTM Model ===")

    # Dummy input (batch_size=4, sequence_length=100)
    dummy_input = np.random.randint(1, 65, size=(4, 100))

    # Our function is build_cnn_bilstm_feature_extractor
    model = build_cnn_bilstm_feature_extractor(vocab_size=65, sequence_length=100)

    # This prints model summary
    model.summary()

    # Forward pass
    output = model.predict(dummy_input)

    print("\nOutput shape:", output.shape)
