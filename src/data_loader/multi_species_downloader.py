import os
import time
import ssl
import http.client
from Bio import Entrez

# Fix SSL certificate verify failed issue on Windows
ssl._create_default_https_context = ssl._create_unverified_context
http.client._MAXHEADERS = 1000

Entrez.email = "vishnuvarthansenthilkumar@gmail.com"

SPECIES_LIST = [
    "Gallus gallus",
    "Anas platyrhynchos",
    "Meleagris gallopavo",
    "Anser anser",
    "Columba livia",
    "Passer domesticus"
]

# We search specific Avian Influenza strains
STRAINS = ["H5N1", "H5N2", "H5N6", "H5N8", "H7N3", "H7N7", "H7N9", "H9N2"]

def search_ncbi(query, max_results=2000, batch_size=100):
    print(f"\nSearching NCBI for: '{query}' (target max: {max_results})....")
    query_ids = []
    seen_ids = set()
    
    try:
        for start_index in range(0, max_results, batch_size):
            fetch_size = min(batch_size, max_results - start_index)
            handle = Entrez.esearch(db="nucleotide", term=query, retstart=start_index, retmax=fetch_size)
            record = Entrez.read(handle)
            handle.close()
            
            ids = record.get("IdList", [])
            if not ids:
                break
                
            unique_ids = [seq_id for seq_id in ids if seq_id not in seen_ids]
            if unique_ids:
                query_ids.extend(unique_ids)
                seen_ids.update(unique_ids)
            
            print(f"  -- Fetched IDs {start_index} to {start_index + len(ids)} (Unique in batch: {len(unique_ids)}, Total: {len(query_ids)})...")
            time.sleep(1.5)
            
            if len(query_ids) >= max_results:
                break
                
        return query_ids[:max_results]
    except Exception as e:
        print(f"Error searching NCBI for '{query}': {e}")
        return query_ids

def fetch_fasta(ids, batch_size=50):
    if not ids:
        return ""
    
    print(f"Fetching {len(ids)} sequences from NCBI in batches of {batch_size}...")
    fasta_data_list = []
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        print(f"  -- Downloading batch {i} to {i+len(batch_ids)}...")
        
        retries = 3
        success = False
        while retries >= 0 and not success:
            try:
                handle = Entrez.efetch(db="nucleotide", id=batch_ids, rettype="fasta", retmode="text")
                fasta_data = handle.read()
                handle.close()
                fasta_data_list.append(fasta_data)
                success = True
            except Exception as e:
                print(f"     [!] Fetch failed for batch {i}: {e}")
                if retries > 0:
                    print(f"     [*] Retrying in 5s... ({retries} left)")
                    time.sleep(5)
                retries -= 1
        
        if not success:
            print(f"     [!] Skipping batch {i} to {i+len(batch_ids)} after failures.")
        time.sleep(1.5)
            
    return "\n".join(fasta_data_list)

def save_fasta(sequences, output_path):
    if not sequences or not sequences.strip():
        print(f"No sequences to save for '{output_path}'.")
        return
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(sequences + "\n")
        sequence_count = sequences.count(">")
        print(f"Successfully appended {sequence_count} sequences to {output_path}")
    except Exception as e:
        print(f"Error saving sequences to {output_path}: {e}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    
    for species in SPECIES_LIST:
        print(f"\n=======================================================")
        print(f" PROCESSING SPECIES: {species}")
        print(f"=======================================================")
        
        safe_species_name = species.lower().replace(" ", "_")
        species_dir = os.path.join(base_dir, safe_species_name)
        
        # 1. Healthy Query
        healthy_query = f"\"{species}\"[Organism] AND (cds[Feature Key] OR mRNA[Filter]) NOT (virus[Organism] OR influenza[All Fields])"
        healthy_file = os.path.join(species_dir, "healthy.fasta")
        
        # We target a max of 2000 healthy per species to keep downloads reasonable, except Gallus gallus
        target_healthy = 10000 if species == "Gallus gallus" else 3000
        
        ids_healthy = search_ncbi(healthy_query, max_results=target_healthy)
        if ids_healthy:
            fasta_data = fetch_fasta(ids_healthy)
            save_fasta(fasta_data, healthy_file)
            
        # 2. Infected Queries
        target_infected_per_strain = 1250 if species == "Gallus gallus" else 400
        infected_file = os.path.join(species_dir, "infected.fasta")
        
        for strain in STRAINS:
            query = f"{strain} Avian Influenza AND \"{species}\"[Organism]"
            ids_infected = search_ncbi(query, max_results=target_infected_per_strain)
            if ids_infected:
                fasta_data = fetch_fasta(ids_infected)
                save_fasta(fasta_data, infected_file)

if __name__ == "__main__":
    main()
