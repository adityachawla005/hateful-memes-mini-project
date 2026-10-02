"""
Shared utilities for the ICT-4442 Hateful Memes comparison.

Every model script (clip_classifier.py, mlp_fusion.py, cnn_lstm.py, visualbert.py)
imports from here so that the dataset split and the evaluation metrics are IDENTICAL
across all four models. The comparison is the graded deliverable, so nothing about the
data or the scoring is allowed to differ between members.

Split policy (stated in the report):
    train    = train.jsonl        -> training
    dev_seen = dev_seen.jsonl     -> validation / model selection (early stopping)
    test_seen= test_seen.jsonl    -> final reported metrics
The official `test_unseen` labels are withheld, so they are NOT used for scoring.
"""
import os
import json
import random

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths — defaults match the Kaggle "williamberrios/hateful-memes" mirror used
# for development. Override with the HM_BASE environment variable if your copy
# lives elsewhere (e.g. a local ./data/hateful_memes).
# --------------------------------------------------------------------------- #
BASE = os.environ.get(
    "HM_BASE",
    "/kaggle/input/datasets/williamberrios/hateful-memes/hateful_memes",
)
WORK_DIR = os.environ.get("HM_WORK", "/kaggle/working")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_SPLIT = "train"
VAL_SPLIT = "dev_seen"     # labelled -> model selection
TEST_SPLIT = "test_seen"   # labelled -> reported
EVAL_SPLITS = [VAL_SPLIT, TEST_SPLIT]

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_split(split: str, base: str = BASE) -> pd.DataFrame:
    """Read one .jsonl split into a DataFrame with columns id, img, text, label."""
    path = os.path.join(base, f"{split}.jsonl")
    df = pd.read_json(path, lines=True)
    return df


def image_path(img_field: str, base: str = BASE) -> str:
    """Resolve an 'img' field (e.g. 'img/12345.png') to an absolute path."""
    p = os.path.join(base, img_field)
    if os.path.exists(p):
        return p
    # fallback: some mirrors drop the leading img/ folder
    return os.path.join(base, "img", os.path.basename(img_field))


# --------------------------------------------------------------------------- #
# The ONE shared metric function. Primary = ROC-AUC. Supplementary = acc/P/R/F1.
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, split_name: str = "") -> dict:
    """y_prob = P(hateful). Threshold 0.5 for the discrete metrics."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_recall_fscore_support,
    )
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)

    m = {"split": split_name, "roc_auc": auc, "accuracy": acc,
         "precision": float(p), "recall": float(r), "f1": float(f1)}
    if split_name:
        print(f"[{split_name}] AUC={auc:.4f}  Acc={acc:.4f}  "
              f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    return m


def save_results(model_name: str, results: list) -> str:
    """Persist a list of per-split metric dicts to results/<model_name>_results.json."""
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {path}")
    return path
