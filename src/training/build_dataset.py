import os
import sys
import json
import csv
import numpy as np
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Bio import SeqIO
from sklearn.model_selection import train_test_split

from src.preprocessing.dna_preprocessing import clean_dna, validate_dna
from src.representation.kmer_representation import build_vocab, generate_kmers, encode_kmers, pad_encoded_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetStats:
    def __init__(self):
        self.total_downloaded = 0
        self.total_healthy = 0
        self.total_infected = 0
        self.duplicates_removed = 0
        self.invalids_removed = 0
        self.lengths = []
        self.species_counts = {}

    def add_species(self, species_name):
        self.species_counts[species_name] = self.species_counts.get(species_name, 0) + 1

stats = DatasetStats()
seen_sequences = set()

def parse_fasta_robust(file_path):
    records = []
    current_header = None
    current_seq = []
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith(";") or line_str.startswith("#"):
                    continue
                if line_str.startswith(">"):
                    if current_header:
                        records.append((current_header, "".join(current_seq)))
                    current_header = line_str[1:]
                    current_seq = []
                else:
                    current_seq.append(line_str)
            if current_header:
                records.append((current_header, "".join(current_seq)))
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        
    return records

def process_fasta_files(directory_path, label, k=3, max_length=500, vocab=None, is_healthy=True):
    X = []
    y = []
    
    path = Path(directory_path)
    if not path.exists():
        logging.warning(f"Directory {directory_path} does not exist.")
        return X, y
        
    fasta_files = list(path.rglob("*.fasta")) + list(path.rglob("*.fa"))
    
    for file_path in fasta_files:
        logging.info(f"Processing: {file_path.name}")
        
        species_name = file_path.stem.split('_')[0].upper() 
        
        records = parse_fasta_robust(file_path)
        stats.total_downloaded += len(records)
        
        for header, raw_seq in records:
            cleaned = clean_dna(raw_seq)
            
            # Check mapping viability
            if not validate_dna(cleaned) or len(cleaned) < k:
                stats.invalids_removed += 1
                continue
                
            # Deduplication check
            if cleaned in seen_sequences:
                stats.duplicates_removed += 1
                continue
            
            seen_sequences.add(cleaned)
            stats.add_species(species_name)
            stats.lengths.append(len(cleaned))
            
            if is_healthy:
                stats.total_healthy += 1
            else:
                stats.total_infected += 1
            
            # K-mers Extract and Encode
            kmers = generate_kmers(cleaned, k)
            encoded = encode_kmers(kmers, vocab)
            padded = pad_encoded_sequence(encoded, max_length)
            
            X.append(padded)
            y.append(label)
            
    return X, y

def balance_dataset(X_h, y_h, X_i, y_i):
    """ Return the dataset with its natural distribution without oversampling or downsampling. """
    len_h = len(X_h)
    len_i = len(X_i)
    
    logging.info(f"Natural class distribution - Healthy: {len_h}, Infected: {len_i}")
    
    return X_h + X_i, y_h + y_i

def build_dataset_pipeline():
    logging.info("Starting Dataset Build Pipeline...")
    project_root = Path(__file__).resolve().parent.parent.parent
    
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    metadata_dir = project_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    healthy_dir = raw_dir / "healthy"
    infected_dir = raw_dir / "infected"
    
    # Sequence Architecture Defaults
    K_LENGTH = 3
    MAX_SEQ_LENGTH = 500
    
    vocab = build_vocab(K_LENGTH)
    
    # Extractor
    X_healthy, y_healthy = process_fasta_files(healthy_dir, label=0, k=K_LENGTH, max_length=MAX_SEQ_LENGTH, vocab=vocab, is_healthy=True)
    X_infected, y_infected = process_fasta_files(infected_dir, label=1, k=K_LENGTH, max_length=MAX_SEQ_LENGTH, vocab=vocab, is_healthy=False)
    
    # Balance Dataset
    X_all, y_all = balance_dataset(X_healthy, y_healthy, X_infected, y_infected)
    
    if not X_all:
        logging.error("No sequences mapped successfully.")
        return
        
    X_matrix = np.array(X_all)
    y_matrix = np.array(y_all)
    
    logging.info(f"Balanced Matrix Dimensions: {X_matrix.shape}")
    
    # 70/15/15 Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_matrix, y_matrix, test_size=0.15, random_state=42, stratify=y_matrix
    )
    
    # Validation is 15% of total, which is 15/85 of temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )
    
    # Export endpoints
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(train_dir / "X_train.npy", X_train)
    np.save(train_dir / "y_train.npy", y_train)
    np.save(val_dir / "X_val.npy", X_val)
    np.save(val_dir / "y_val.npy", y_val)
    np.save(test_dir / "X_test.npy", X_test)
    np.save(test_dir / "y_test.npy", y_test)
    
    # Generate Reports
    stats_dict = {
        "Total Downloaded Sequences Processed": stats.total_downloaded,
        "Total Healthy Samples Found": stats.total_healthy,
        "Total Infected Samples Found": stats.total_infected,
        "Total Final Balanced Dataset Size": len(X_all),
        "Species / Strain Distribution": stats.species_counts,
        "Average Sequence Length": float(np.mean(stats.lengths)) if stats.lengths else 0,
        "Minimum Sequence Length": int(np.min(stats.lengths)) if stats.lengths else 0,
        "Maximum Sequence Length": int(np.max(stats.lengths)) if stats.lengths else 0,
        "Number of Removed Duplicates": stats.duplicates_removed,
        "Number of Discarded Invalid Sequences": stats.invalids_removed
    }
    
    # Save JSON
    with open(metadata_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    # Save CSV
    with open(metadata_dir / "dataset_statistics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats_dict.items():
            if k == "Species / Strain Distribution": continue
            writer.writerow([k, v])
            
    logging.info("Exported Processed Dataset Matrices and Statistics Successfully.")

if __name__ == "__main__":
    build_dataset_pipeline()
