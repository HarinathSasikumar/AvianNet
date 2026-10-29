# Avian DNA Classifier

## Project Description
The **Avian DNA Classifier** is a machine learning project focused on classifying DNA sequences related to avian influenza. This project explores genetic sequence representations to build classification models that identify and categorize specific strains and sequences.

## Folder Structure
- `data/`: Contains datasets (raw, interim, processed)
- `notebooks/`: Jupyter notebooks for data exploration and rapid prototyping
- `src/`: Source code including scripts for data logic and pipeline
  - `preprocessing/`: Code for sequence cleaning and preparation
  - `representation/`: Feature engineering and sequence representation (e.g., k-mers, embeddings)
  - `models/`: Machine learning model architectures
  - `training/`: Training routines and pipelines
  - `evaluation/`: Scripts to evaluate model metrics
  - `utils/`: Common utility functions
- `outputs/`: Saved models, logs, metrics, figures, and plots
- `configs/`: Configuration files (e.g., hyperparameter configurations)

## Setup Instructions

### 1. Set Up Environment
Create and activate a virtual environment to manage dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Initialize Project Structure
Run the setup script (located in the parent directory or root level) if any directories are missing:
```bash
python setup_project.py
```

### 4. Run Application
Verify system initialization by running `main.py`:
```bash
python main.py
```
