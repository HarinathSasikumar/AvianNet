import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

project_root = Path(__file__).resolve().parent.parent.parent

def plot_training():
    history_path = project_root / "training_history.json"
    
    if not history_path.exists():
        logging.error(f"Cannot find {history_path}. Has train_model.py been run successfully yet?")
        return
        
    logging.info(f"Loading history from {history_path}...")
    # Load history
    with open(history_path, "r") as f:
        history = json.load(f)

    # Extract values
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    accuracy = history.get("accuracy", [])
    val_accuracy = history.get("val_accuracy", [])

    # --- LOSS GRAPH ---
    plt.figure()
    plt.plot(loss, label="Training Loss", color='blue')
    plt.plot(val_loss, label="Validation Loss", color='orange')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    loss_plot_path = project_root / "loss_plot.png"
    plt.savefig(loss_plot_path)
    logging.info(f"Saved {loss_plot_path}")
    plt.show()

    # --- ACCURACY GRAPH ---
    plt.figure()
    plt.plot(accuracy, label="Training Accuracy", color='blue')
    plt.plot(val_accuracy, label="Validation Accuracy", color='orange')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    
    acc_plot_path = project_root / "accuracy_plot.png"
    plt.savefig(acc_plot_path)
    logging.info(f"Saved {acc_plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_training()
