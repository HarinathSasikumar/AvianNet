import os
import glob
import json
import random
import csv
from collections import defaultdict

# Re-use existing validation logic
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.dna_preprocessing import clean_dna, validate_dna

def parse_fasta(filepath, species, label):
    """Yields (seq_id, description, sequence, species, label) from a FASTA file."""
    if not os.path.exists(filepath):
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        seq_id = ""
        desc = ""
        seq_lines = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    yield (seq_id, desc, "".join(seq_lines), species, label)
                
                parts = line[1:].split(" ", 1)
                seq_id = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
                seq_lines = []
            else:
                seq_lines.append(line)
                
        if seq_id:
            yield (seq_id, desc, "".join(seq_lines), species, label)

def main():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    meta_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata"))
    
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # 1. Parsing & Validation
    seen_sequences = set()
    seen_ids = set()
    
    valid_data = []
    
    stats = {
        "total_parsed": 0,
        "invalid_removed": 0,
        "duplicates_removed": 0,
        "final_valid": 0,
        "species_breakdown": defaultdict(lambda: {"Healthy": 0, "Infected": 0, "total": 0})
    }
    
    print("=== Processing Multi-Species DNA Data ===")
    
    species_folders = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for sf in species_folders:
        species_name = sf.replace("_", " ").capitalize()
        healthy_file = os.path.join(raw_dir, sf, "healthy.fasta")
        infected_file = os.path.join(raw_dir, sf, "infected.fasta")
        
        for file_path, label in [(healthy_file, 0), (infected_file, 1)]:
            if not os.path.exists(file_path):
                continue
                
            for seq_id, desc, raw_seq, sp, lbl in parse_fasta(file_path, species_name, label):
                stats["total_parsed"] += 1
                
                # Deduplicate by exact accession ID first
                if seq_id in seen_ids:
                    stats["duplicates_removed"] += 1
                    continue
                seen_ids.add(seq_id)
                
                cleaned_seq = clean_dna(raw_seq)
                
                if not validate_dna(cleaned_seq) or len(cleaned_seq) < 50:
                    stats["invalid_removed"] += 1
                    continue
                    
                # Deduplicate by sequence content
                if cleaned_seq in seen_sequences:
                    stats["duplicates_removed"] += 1
                    continue
                seen_sequences.add(cleaned_seq)
                
                valid_data.append({
                    "id": seq_id,
                    "desc": desc,
                    "sequence": cleaned_seq,
                    "length": len(cleaned_seq),
                    "species": sp,
                    "label": lbl
                })
                
                lbl_str = "Infected" if lbl == 1 else "Healthy"
                stats["species_breakdown"][sp][lbl_str] += 1
                stats["species_breakdown"][sp]["total"] += 1

    stats["final_valid"] = len(valid_data)
    
    # 2. Saving Metadata CSV
    csv_path = os.path.join(meta_dir, "sequence_metadata.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "species", "infection_status", "sequence_length", "description"])
        for item in valid_data:
            status = "Infected" if item["label"] == 1 else "Healthy"
            writer.writerow([item["id"], item["species"], status, item["length"], item["desc"]])
            
    # 3. Saving Quality Report
    report_path = os.path.join(meta_dir, "data_quality_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    # 4. Generating Train/Val/Test Splits (Stratified across species where possible)
    random.seed(42)
    random.shuffle(valid_data)
    
    train_data, val_data, test_data = [], [], []
    
    # Stratify by species and label
    by_category = defaultdict(list)
    for item in valid_data:
        by_category[(item["species"], item["label"])].append(item)
        
    for cat, items in by_category.items():
        n = len(items)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train+n_val])
        test_data.extend(items[n_train+n_val:])
        
    # Save splits as JSONL for the pipeline
    for split_name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(proc_dir, f"multi_species_{split_name}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
                
    print("\n=== Data Processing Complete ===")
    print(f"Total sequences parsed:     {stats['total_parsed']}")
    print(f"Invalid sequences removed:  {stats['invalid_removed']}")
    print(f"Duplicates removed:         {stats['duplicates_removed']}")
    print(f"Final valid sequences:      {stats['final_valid']}")
    
    print("\n--- Species Distribution ---")
    for sp, s in stats["species_breakdown"].items():
        print(f"{sp:<22} | Healthy: {s['Healthy']:<5} | Infected: {s['Infected']:<5} | Total: {s['total']}")

if __name__ == "__main__":
    main()
