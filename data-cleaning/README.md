# data-cleaning

Python scripts for data ingestion and preprocessing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python clean_data.py
```

Produces `output/clean_data.csv`, consumed by `../r-models` (or a Python
training script) for model training.
