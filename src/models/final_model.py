import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Multiply,
    Lambda, Flatten, Activation, RepeatVector,
    Permute, Dropout
)
import numpy as np


def _build_shared_layers(vocab_size: int, sequence_length: int):
    """
    Internal helper: builds all shared layers (Embedding → CNN → BiLSTM → Attention → Head)
    and returns the (inputs, prediction, attention_weights) tensors.
    This is shared by both the training model and the inference model.
    """
    inputs = Input(shape=(sequence_length,), name="sequence_input")

    # 1. Embedding Layer
    x = Embedding(input_dim=vocab_size + 1, output_dim=64, name="kmer_embedding")(inputs)

    # 2. CNN Spatial Feature Extraction
    x = Conv1D(filters=128, kernel_size=3, activation='relu', name="conv1d_1")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)

    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool_2")(x)

    # 3. BiLSTM Temporal Sequence Context
    bilstm_out = Bidirectional(LSTM(units=64, return_sequences=True), name="bilstm_block")(x)

    # 4. Attention Mechanism
    attention_scores = Dense(1, activation='tanh', name="attention_dense")(bilstm_out)
    attention_scores = Flatten(name="attention_flatten")(attention_scores)
    attention_weights = Activation('softmax', name="attention_softmax")(attention_scores)

    # Broadcast attention weights across all 128 BiLSTM feature dimensions
    attention_rescaled = RepeatVector(128)(attention_weights)
    attention_rescaled = Permute([2, 1])(attention_rescaled)

    attended_sequence = Multiply(name="attention_multiply")([bilstm_out, attention_rescaled])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1), name="context_vector")(attended_sequence)

    # 5. Feature Dense Layer
    features = Dense(units=128, activation='relu', name="feature_vector")(context_vector)

    # 6. Classification Head
    x = Dense(units=64, activation='relu', name="dense_64_classifier")(features)
    x = Dropout(0.3, name="dropout_regularization")(x)
    prediction = Dense(units=1, activation='sigmoid', name="prediction_output")(x)

    return inputs, prediction, attention_weights


def build_training_model(vocab_size: int, sequence_length: int) -> Model:
    """
    Builds the TRAINING model — single output: prediction only.
    Use this during model.fit() to avoid the Keras multi-output/single-label KeyError.

    Args:
        vocab_size (int): Size of the k-mer vocabulary (e.g. 64 for k=3).
        sequence_length (int): Fixed padded length of input sequences.

    Returns:
        tf.keras.Model: Compiled model with output shape (batch, 1).
    """
    inputs, prediction, _ = _build_shared_layers(vocab_size, sequence_length)

    model = Model(inputs=inputs, outputs=prediction, name="avian_training_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def build_inference_model(vocab_size: int, sequence_length: int) -> Model:
    """
    Builds the INFERENCE model — dual output: [prediction, attention_weights].
    Use this after training (load weights from training model) to get
    both the classification result AND the explainable attention map.

    Args:
        vocab_size (int): Size of the k-mer vocabulary (e.g. 64 for k=3).
        sequence_length (int): Fixed padded length of input sequences.

    Returns:
        tf.keras.Model: Model with outputs [prediction (batch,1), attention_weights (batch, T)].
    """
    inputs, prediction, attention_weights = _build_shared_layers(vocab_size, sequence_length)

    model = Model(
        inputs=inputs,
        outputs=[prediction, attention_weights],
        name="avian_inference_model"
    )

    return model


# --- BACKWARDS COMPATIBILITY ALIAS ---
# Keep build_final_model() working so test_pipeline.py doesn't break
def build_final_model(vocab_size: int, sequence_length: int) -> Model:
    """
    Legacy alias → returns the inference model (prediction + attention).
    For training, use build_training_model() instead.
    """
    return build_inference_model(vocab_size, sequence_length)


if __name__ == "__main__":
    print("=== Testing Training Model (single output) ===")
    dummy_input = np.random.randint(0, 65, size=(4, 100))

    train_model = build_training_model(vocab_size=64, sequence_length=100)
    train_model.summary()
    preds = train_model.predict(dummy_input)
    print(f"\nTraining model output shape: {preds.shape}")  # Expected: (4, 1)

    print("\n=== Testing Inference Model (dual output) ===")
    infer_model = build_inference_model(vocab_size=64, sequence_length=100)
    predictions, attention = infer_model.predict(dummy_input)
    print(f"Inference model prediction shape: {predictions.shape}")  # Expected: (4, 1)
    print(f"Inference model attention shape:  {attention.shape}")    # Expected: (4, T)
