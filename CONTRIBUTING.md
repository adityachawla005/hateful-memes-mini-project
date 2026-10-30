# Contributing — individual ownership & commit history

Phase 2 requires a GitHub repo whose **commit history evidences individual contribution**.
To make that legible, each member owns one model file and commits it under their own git
identity.

| Member | Reg. No. | Owns | Files |
|---|---|---|---|
| Parth Khator | 230911422 | MLP feature fusion | `mlp_fusion.py` |
| Sarthak Atrish | 230911504 | CNN–LSTM fusion | `cnn_lstm.py` |
| Aditya Chawla | 230911260 | VisualBERT | `visualbert.py` |
| Sahithya Sambhrama | 230911014 | CLIP classifier | `clip_classifier.py` |

Shared files (`common.py`, `README.md`, configs) are co-owned — whoever sets up the repo
commits them first.

### Set your identity before committing (do this once per member, per machine)

```bash
# use your own learner email; replace with the correct address if different
git config user.name  "Your Name"
git config user.email "your-regno@learner.manipal.edu"
```

### Suggested first commits (so the graph shows four distinct contributors)

```bash
# whoever bootstraps the repo (e.g. Sahithya)
git add README.md common.py CONTRIBUTING.md .gitignore requirements.txt make_results_table.py
git commit -m "Project scaffold: shared data split + metric protocol"

# Sahithya Sambhrama (230911014) — CLIP
git add clip_classifier.py results/clip_results.json results/clip_large_results.json
git commit -m "CLIP classifier: frozen ViT-L/14 embeddings + MLP head; test_seen ROC-AUC 0.822"

# Parth Khator (230911422) — MLP fusion
git add mlp_fusion.py results/mlp_fusion_results.json
git commit -m "MLP feature fusion: mean-pooled BERT + ResNet-50 with interaction term; 0.700"

# Sarthak Atrish (230911504) — CNN-LSTM
git add cnn_lstm.py results/cnn_lstm_results.json
git commit -m "CNN-LSTM fusion: frozen ResNet-18 + BiLSTM over GloVe-200d; 0.682"

# Aditya Chawla (230911260) — VisualBERT
git add visualbert.py
git commit -m "VisualBERT over torchvision region features (no detectron2)"
```

Keep committing as you iterate (feature extraction, tuning, bug fixes) — small frequent
commits per member are exactly the evidence the viva looks for.

### Verify the contribution split

```bash
git shortlog -sne        # commits per author
```
