import os
import sys
import time
import ssl
import http.client
from Bio import Entrez

# Fix SSL certificate verify failed issue which is common on Windows
ssl._create_default_https_context = ssl._create_unverified_context

# Increase max headers line length if needed for long HTTP headers
http.client._MAXHEADERS = 1000

# Set Entrez email (required by NCBI)
Entrez.email = "vishnuvarthansenthilkumar@gmail.com"

# Global set to track duplicate sequence IDs
seen_ids = set()

def search_ncbi(query, max_results=5000, batch_size=100):
    """
    Search NCBI nucleotide database with pagination and return a list of unique sequence IDs.
    """
    print(f"\nSearching NCBI for: '{query}' (target max: {max_results})....")
    query_ids = []
    
    try:
        for start_index in range(0, max_results, batch_size):
            fetch_size = min(batch_size, max_results - start_index)
            
            handle = Entrez.esearch(db="nucleotide", term=query, retstart=start_index, retmax=fetch_size)
            record = Entrez.read(handle)
            handle.close()
            
            ids = record.get("IdList", [])
            if not ids:
                break
                
            # Filter duplicates globally
            unique_ids = [seq_id for seq_id in ids if seq_id not in seen_ids]
            
            if unique_ids:
                query_ids.extend(unique_ids)
                seen_ids.update(unique_ids)
            
            print(f"  -- Fetched IDs {start_index} to {start_index + len(ids)} (Unique in batch: {len(unique_ids)}, Total accumulated: {len(query_ids)})...")
            time.sleep(2) # Be polite to NCBI servers
            
            # Break early if we've accumulated enough total unique IDs for this query
            if len(query_ids) >= max_results:
                break
                
        return query_ids[:max_results]
    except Exception as e:
        print(f"Error searching NCBI for '{query}': {e}")
        return query_ids

def fetch_fasta(ids, batch_size=50):
    """
    Fetch sequences in FASTA format for a list of NCBI sequence IDs.
    Batching fetches to prevent overloading NCBI servers.
    """
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
                    print(f"     [*] Retrying in 5 seconds... ({retries} left)")
                    time.sleep(5)
                retries -= 1
        
        if not success:
            print(f"     [!] Skipping batch {i} to {i+len(batch_ids)} after multiple failures.")
            
        time.sleep(2) # Delay between fetches
            
    return "\n".join(fasta_data_list)

def save_fasta(sequences, output_path):
    """
    Save fetched FASTA sequences to a file.
    """
    if not sequences or not sequences.strip():
        print(f"No sequences provided to save for '{output_path}'.")
        return
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sequences)
            
        sequence_count = sequences.count(">")
        print(f"Successfully saved {sequence_count} sequences to {output_path}")
    except Exception as e:
        print(f"Error saving sequences to {output_path}: {e}")

def main():
    # 1. Healthy Query: Strictly Gallus gallus cds (Chicken) ONLY
    healthy_queries = {
        "Gallus gallus cds": "data/raw/healthy/general/gallus_gallus_cds_expanded.fasta"
    }

    # 2. Infected Queries: Avian Influenza strains strictly for Gallus gallus
    infected_queries = {
        "H5N1 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H5/H5N1_expanded.fasta",
        "H5N2 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H5/H5N2_expanded.fasta",
        "H5N6 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H5/H5N6_expanded.fasta",
        "H5N8 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H5/H5N8_expanded.fasta",
        "H7N3 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H7/H7N3_expanded.fasta",
        "H7N7 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H7/H7N7_expanded.fasta",
        "H7N9 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H7/H7N9_expanded.fasta",
        "H9N2 Avian Influenza AND \"Gallus gallus\"[Organism]": "data/raw/infected/H9/H9N2_expanded.fasta"
    }
    
    # Target 10,000 sequences for Healthy (Gallus gallus cds)
    print("=== BEGINNING HEALTHY DOWNLOADS (Gallus gallus cds ONLY) ===")
    for query, output_path in healthy_queries.items():
        ids = search_ncbi(query, max_results=10000)
        if ids:
            fasta_data = fetch_fasta(ids)
            save_fasta(fasta_data, output_path)

    # Target 1,250 per query for Infected to total ~10,000 sequences
    print("\n=== BEGINNING INFECTED DOWNLOADS ===")
    for query, output_path in infected_queries.items():
        ids = search_ncbi(query, max_results=1250)
        if ids:
            fasta_data = fetch_fasta(ids)
            save_fasta(fasta_data, output_path)

if __name__ == "__main__":
    main()
