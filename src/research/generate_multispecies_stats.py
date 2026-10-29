import json
import csv
import numpy as np
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    meta_file = root / "data" / "metadata" / "sequence_metadata.csv"
    lengths = []
    species_counts = {}
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            l = int(row['sequence_length'])
            lengths.append(l)
            sp = row['species']
            st = row['infection_status']
            if sp not in species_counts:
                species_counts[sp] = {'Healthy': 0, 'Infected': 0, 'total': 0}
            species_counts[sp][st] += 1
            species_counts[sp]['total'] += 1

    lengths = np.array(lengths)

    # 70% train, 15% val, 15% test stratified
    splits = {'train': 0, 'val': 0, 'test': 0, 'by_species': {}}
    for sp, data in species_counts.items():
        sp_tr, sp_va, sp_te = 0, 0, 0
        for st in ['Healthy', 'Infected']:
            cnt = data[st]
            tr = int(cnt * 0.7)
            va = int(cnt * 0.15)
            te = cnt - tr - va
            sp_tr += tr
            sp_va += va
            sp_te += te
        splits['train'] += sp_tr
        splits['val'] += sp_va
        splits['test'] += sp_te
        splits['by_species'][sp] = {'train': sp_tr, 'val': sp_va, 'test': sp_te}

    stats = {
        'total_sequences': int(len(lengths)),
        'total_healthy': int(sum(s['Healthy'] for s in species_counts.values())),
        'total_infected': int(sum(s['Infected'] for s in species_counts.values())),
        'mean_length': round(float(np.mean(lengths)), 2),
        'median_length': round(float(np.median(lengths)), 2),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'std_length': round(float(np.std(lengths)), 2),
        'species_distribution': species_counts,
        'splits': splits
    }

    out_file = root / "data" / "metadata" / "species_statistics.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
