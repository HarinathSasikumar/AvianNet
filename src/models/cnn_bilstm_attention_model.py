import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Multiply, Reshape, Lambda, Flatten, Activation, RepeatVector, Permute
import numpy as np

def build_cnn_bilstm_attention_extractor(vocab_size: int, sequence_length: int) -> Model:
    """
    Builds a CNN + BiLSTM model featuring an Attention Mechanism.
    Highlights exactly which sequential features/motifs contribute most to the final state.
    
    Args:
        vocab_size (int): Size of the k-mer vocabulary (+1).
        sequence_length (int): Fixed chronological sequence length.
        
    Returns:
        tf.keras.Model: Returns both the dense features AND the local attention scores.
    """
    inputs = Input(shape=(sequence_length,), name="sequence_input")
    
    # 1. Embedding Layer
    x = Embedding(input_dim=vocab_size, output_dim=64, name="kmer_embedding")(inputs)
    
    # 2. CNN Spatial Logic
    x = Conv1D(filters=128, kernel_size=3, activation='relu', name="conv1d_1")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_2")(x)
    
    # 3. Sequence Intelligence Block
    # return_sequences=True gives us a 3D matrix (batch, timesteps, features)
    bilstm_out = Bidirectional(LSTM(units=64, return_sequences=True), name="bilstm_block")(x)
    
    # 4. Attention Mechanism 🔥 (Keras Correct Topology)
    # Get raw scores
    attention_scores = Dense(1, activation='tanh', name="attention_dense")(bilstm_out)
    attention_scores = Flatten(name="attention_flatten")(attention_scores)
    # Apply softmax on time dimension to get sum=1 probabilities
    attention_weights = Activation('softmax', name="attention_softmax")(attention_scores)
    
    # Repeat the 1D scores onto all 128 feature axes
    # bilstm_out shape is (None, timesteps, 128)
    attention_rescaled = RepeatVector(128)(attention_weights)
    attention_rescaled = Permute([2, 1])(attention_rescaled)
    
    # Multiply structurally
    attended_sequence = Multiply(name="attention_multiply")([bilstm_out, attention_rescaled])
    
    # Securely Reduce Sum via Lambda Layer explicitly compliant with Keras API
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1), name="context_vector")(attended_sequence)
    
    # 5. Dense Latent Output
    features = Dense(units=128, activation='relu', name="feature_vector")(context_vector)
    
    # Connect Model outputs functionally wrapping BOTH targets
    model = Model(inputs=inputs, outputs=[features, attention_weights], name="cnn_bilstm_attention_dna_extractor")
    return model

if __name__ == "__main__":
    print("=== Testing CNN + BiLSTM + Attention Model ===")
    dummy_input = np.random.randint(1, 65, size=(4, 100))

    model = build_cnn_bilstm_attention_extractor(vocab_size=65, sequence_length=100)

    # 1. Verification Printout
    model.summary()

    # 2. Forward Propagation Check
    features, attention = model.predict(dummy_input)

    print("\n[Architectural Compliance Checks]")
    print(f"Feature vector shape: {features.shape}")
    print(f"Attention shape: {attention.shape}")
