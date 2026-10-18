"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATALOG_PATH = DATA_DIR / "product_catalog.csv"
PLANOGRAM_PATH = DATA_DIR / "expected_planogram.json"
SAMPLES_DIR = DATA_DIR / "sample_shelves"

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"

# Parameter counts (billions) of vision SLMs, used for the "SLM < 20B" badge.
MODEL_PARAMS_B = {
    "qwen2.5vl:3b": 3,
    "qwen2.5vl:7b": 7,
    "llama3.2-vision:11b": 11,
    "minicpm-v:8b": 8,
    "moondream:1.8b": 1.8,
    "gemma3:4b": 4,
}

# Parameter-count threshold for the hackathon's SLM bonus.
SLM_LIMIT_B = 20

# Shelf-label price tolerance (INR).
PRICE_TOLERANCE = 0.5

# Share of a SKU's sales assumed lost for each violation type.
LOSS_FACTORS = {"EMPTY": 1.0, "LOW": 0.35, "MISPLACED": 0.5, "UNKNOWN": 0.5, "LABEL": 0.1}

# Revenue-at-risk severity thresholds (INR/day).
CRITICAL_AT = 500
HIGH_AT = 150
