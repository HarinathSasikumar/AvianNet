import os
import numpy as np
import logging
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.final_model import build_training_model, build_inference_model
from src.representation.kmer_representation import build_vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_pipeline():
    logging.info("Starting Evaluation Diagnostics...")
    
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "outputs" / "models"
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    try:
        X_test = np.load(processed_dir / "test" / "X_test.npy")
        y_test = np.load(processed_dir / "test" / "y_test.npy")
        logging.info(f"Loaded {len(y_test)} testing sequences natively.")
    except Exception as e:
        logging.error(f"Cannot load testing datasets: {e}")
        return
        
    # 2. Load the best trained model checkpoint
    model_path = models_dir / "best_avian_model.keras"
    if not model_path.exists():
        logging.error(f"Model checkpoint missing at {model_path}. Run train_model.py first!")
        return

    logging.info("Loading saved training model weights...")
    # Load the saved single-output training model first
    saved_training_model = tf.keras.models.load_model(str(model_path))

    # Build the inference model (dual output: prediction + attention)
    # and transfer the shared weights across
    K_LENGTH = 3
    vocab = build_vocab(K_LENGTH)
    vocab_size = len(vocab) + 1
    sequence_length = X_test.shape[1]

    inference_model = build_inference_model(vocab_size=vocab_size, sequence_length=sequence_length)
    inference_model.set_weights(saved_training_model.get_weights())
    logging.info("Weights transferred from training model to inference model successfully.")

    # 3. Run inference — returns [predictions, attention_weights]
    logging.info("Running inference on test set...")
    predictions, attention_maps = inference_model.predict(X_test, verbose=1)

    # Extract flat probability array
    y_pred_prob = predictions.flatten()
    y_pred_classes = (y_pred_prob > 0.5).astype(int)
    
    # 4. Generate Core Statistical Metrics
    logging.info("\n=== Classification Report ===")
    report = classification_report(y_test, y_pred_classes, target_names=["Healthy (0)", "Infected (1)"])
    logging.info(f"\n{report}")
    
    # 5. Build Confusion Matrix Image
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Infected"], yticklabels=["Healthy", "Infected"])
    plt.title("Avian DNA Classification System - Confusion Matrix")
    plt.ylabel('Actual Underlying Label')
    plt.xlabel('Algorithm Prediction')
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Deployed Confusion mapping format natively mapping logic to {cm_path}")
    
    # 6. Build ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Inference Rate')
    plt.ylabel('True Positive Detection Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = figures_dir / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    logging.info(f"Deployed Sequence mapping mapping architecture natively mapping logic to {roc_path}")
    
    logging.info("Diagnostics completely finished. You have successfully arrived at the goal pipeline natively!")

if __name__ == "__main__":
    evaluate_pipeline()
