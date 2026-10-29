import os
from pathlib import Path

def create_directory(path: Path):
    """Creates a directory if it does not exist and prints a message."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {path}")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def create_empty_file(path: Path):
    """Creates an empty file if it does not exist and prints a message."""
    try:
        # Create parent directories just in case they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
            print(f"File created: {path}")
        else:
            print(f"File already exists: {path}")
    except Exception as e:
        print(f"Error creating file {path}: {e}")

def setup_project_structure():
    """Builds the structured dataset directory for the DNA classification project."""
    print("Starting project structure setup...")
    
    # Define directories to create
    directories_to_create = [
        "data/raw/infected/H5",
        "data/raw/infected/H7",
        "data/raw/infected/other",
        "data/raw/healthy/general",
        "data/processed/train",
        "data/processed/test",
        "data/interim",
        "data/metadata",
        "src/data_loader"
    ]
    
    # Create directories
    for dir_path in directories_to_create:
        create_directory(Path(dir_path))
        
    # Define files to create
    files_to_create = [
        "data/metadata/sequence_labels.csv",
        "data/metadata/download_log.csv"
    ]
    
    # Create empty files
    for file_path in files_to_create:
        create_empty_file(Path(file_path))
        
    print("\nProject structure setup complete.")

if __name__ == "__main__":
    setup_project_structure()
