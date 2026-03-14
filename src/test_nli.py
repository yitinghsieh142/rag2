import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MODEL_NAME = os.getenv("NLI_MODEL", "IDEA-CCNL/Erlangshen-Roberta-110M-NLI")
MODEL_NAME = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
_tok = None
_model = None

# -----------------------------
# Lazy load model (只載入一次)
# -----------------------------
def _load_nli():
    global _tok, _model
    if _model is not None:
        return
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()

# -----------------------------
# 安全取得 label 對應機率
# -----------------------------
def _extract_probs(probs_tensor):
    """
    回傳：
    {
      "entailment": float,
      "neutral": float,
      "contradiction": float
    }

    兼容：
    ENTAILMENT / NEUTRAL / CONFLICT / CONTRADICTION
    label 順序不固定
    """
    id2label = _model.config.id2label

    probs = {}
    for i in range(probs_tensor.shape[0]):
        label = str(id2label.get(i, i)).lower()
        probs[label] = float(probs_tensor[i].item())

    # 安全抓取
    p_entail = 0.0
    p_neutral = 0.0
    p_contra = 0.0

    for k, v in probs.items():
        kk = k.lower()
        if "entail" in kk:
            p_entail = v
        elif "neutral" in kk:
            p_neutral = v
        elif "contrad" in kk or "conflict" in kk:
            p_contra = v

    return {
        "entailment": p_entail,
        "neutral": p_neutral,
        "contradiction": p_contra
    }

# -----------------------------
# 核心 API（給 tools.py 用）
# -----------------------------
def compute_nli_score(premise: str, hypothesis: str, max_length: int = 512):
    """
    premise = 檢索文件（全部 context）
    hypothesis = 生成答案

    回傳：
    {
      "entailment": ...,
      "neutral": ...,
      "contradiction": ...,
      "factuality": P_e - P_c,
      "C_fact": 映射到 [0,1]
    }
    """

    if not premise or not hypothesis:
        return {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
            "factuality": 0.0,
            "C_fact": 0.0
        }

    _load_nli()

    x = _tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    with torch.no_grad():
        logits = _model(**x).logits
        probs_tensor = F.softmax(logits, dim=-1)[0]

    probs = _extract_probs(probs_tensor)

    p_e = probs["entailment"]
    p_n = probs["neutral"]
    p_c = probs["contradiction"]

    # Factuality
    C_fact = 1.0 - p_c
    C_fact = max(0.0, min(1.0, C_fact))

    return {
        "entailment": round(p_e, 6),
        "neutral": round(p_n, 6),
        "contradiction": round(p_c, 6),
        "factuality": round(C_fact, 6),
        "C_fact": round(C_fact, 6),
    }