import os
import json
import numpy as np
import logging
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Ensure correct path to the root module
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.final_model import build_training_model
from src.representation.kmer_representation import build_vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pipeline():
    logging.info("Starting ML Training Pipeline...")
    
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs" / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load physical matrices
    try:
        X_train = np.load(processed_dir / "train" / "X_train.npy")
        y_train = np.load(processed_dir / "train" / "y_train.npy")
        X_val = np.load(processed_dir / "test" / "X_test.npy")
        y_val = np.load(processed_dir / "test" / "y_test.npy")
    except FileNotFoundError as e:
        logging.error(f"Required matrices not found in {processed_dir}. Run build_dataset.py first! {e}")
        return
        
    logging.info(f"Loaded Training Sequences: {X_train.shape[0]} arrays of length {X_train.shape[1]}")
    
    # Vocabulary & Sequence Configuration Constraints
    K_LENGTH = 3
    vocab = build_vocab(K_LENGTH)
    vocab_size = len(vocab) + 1  # Add 1 for the '0' padding/unknown value
    sequence_length = X_train.shape[1]
    
    # Instantiate TRAINING model — single output (prediction only)
    # This avoids the Keras KeyError: path (0,) in loss when model has multiple outputs
    model = build_training_model(vocab_size=vocab_size, sequence_length=sequence_length)
    
    # Secure logging & model weights backups
    checkpoint_path = outputs_dir / "best_avian_model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights to handle natural class imbalance without data duplication
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, weights))
    logging.info(f"Computed Class Weights: {class_weight_dict}")
    
    # Fire execution
    logging.info("Firing system hardware fitting loops...")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    import json

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    print("Training history saved!")
    
    logging.info("Execution complete.")
    logging.info(f"Deployed strictly best physical checkpoint logic to {checkpoint_path}")

if __name__ == "__main__":
    train_pipeline()
