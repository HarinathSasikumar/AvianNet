import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_project_structure(project_root: Path) -> bool:
    """
    Verifies that all required folders exist in the project structure.
    """
    required_directories = [
        "data",
        "notebooks",
        "src/preprocessing",
        "src/representation",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "outputs",
        "configs"
    ]
    
    all_exist = True
    for relative_dir in required_directories:
        dir_path = project_root / relative_dir
        if not dir_path.exists() or not dir_path.is_dir():
            logging.error(f"Missing required directory: {dir_path}")
            all_exist = False
            
    return all_exist

def main():
    # Define project root (assuming main.py is in the avian-dna-classifier root directory)
    project_root = Path(__file__).parent.resolve()
    
    # 1. Print startup message
    logging.info("Avian DNA Classification System Initialized")
    
    # 2. Check if all folders exist
    if not check_project_structure(project_root):
        logging.warning("Project structure is incomplete. Please run setup_project.py to create missing folders first.")
    else:
        logging.info("All project folders verified successfully. Ready for ML operations.")

if __name__ == "__main__":
    main()
