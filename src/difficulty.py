import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> rag 2/
MODEL_DIR = str((PROJECT_ROOT / "models" / "query_difficulty").resolve())
_tok = None
_model = None
_device = None

def _load():
    global _tok, _model, _device
    if _model is not None:
        return
    _device = "cpu"
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    _model.eval()

def predict_difficulty(query: str, max_len: int = 128) -> str:
    _load()
    q = (query or "").replace("*", "").strip()
    if not q:
        return "easy"
    x = _tok(q, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    x = {k: v.to(_device) for k, v in x.items()}
    with torch.no_grad():
        pred = int(_model(**x).logits.argmax(-1).item())
    return "hard" if pred == 1 else "easy"