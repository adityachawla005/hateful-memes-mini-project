"""
Collate every results/<model>_results.json into one comparison table (the single table the
report's Results section needs). Reports test_seen metrics per model, sorted by ROC-AUC.

    python make_results_table.py
"""
import os
import glob
import json

import pandas as pd

from common import RESULTS_DIR, TEST_SPLIT

DISPLAY = {
    "cnn_lstm": "CNN–LSTM fusion",
    "mlp_fusion": "MLP feature fusion",
    "visualbert": "VisualBERT",
    "clip": "CLIP (ViT-B/32)",
    "clip_large": "CLIP (ViT-L/14)",
}
ORDER = ["cnn_lstm", "mlp_fusion", "visualbert", "clip", "clip_large"]


def main():
    # Scan both the Kaggle work dir and the repo-local results/ folder, de-duping by name
    # (repo-local copy wins, since that's what gets committed).
    repo_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    paths = {}
    for d in (RESULTS_DIR, repo_results):
        for path in glob.glob(os.path.join(d, "*_results.json")):
            paths[os.path.basename(path)] = path

    rows = []
    for path in paths.values():
        name = os.path.basename(path).replace("_results.json", "")
        with open(path) as fh:
            data = json.load(fh)
        test = next((d for d in data if d.get("split") == TEST_SPLIT), None)
        if test is None:
            continue
        rows.append({
            "key": name,
            "Model": DISPLAY.get(name, name),
            "ROC-AUC": round(test["roc_auc"], 4),
            "Accuracy": round(test["accuracy"], 4),
            "Precision": round(test["precision"], 4),
            "Recall": round(test["recall"], 4),
            "F1": round(test["f1"], 4),
        })
    if not rows:
        print("No results found. Run the model scripts first."); return

    df = pd.DataFrame(rows)
    df["__o"] = df["key"].map({k: i for i, k in enumerate(ORDER)}).fillna(99)
    df = df.sort_values("__o").drop(columns=["key", "__o"])

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    md_path = os.path.join(RESULTS_DIR, "results.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(f"# Comparison on {TEST_SPLIT} (primary metric: ROC-AUC)\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
