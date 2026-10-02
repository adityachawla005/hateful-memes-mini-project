# Contributing — individual ownership & commit history

Phase 2 requires a GitHub repo whose **commit history evidences individual contribution**.
To make that legible, each member owns one model file and commits it under their own git
identity.

| Member | Owns | Files |
|---|---|---|
| Member 1 | MLP feature fusion | `mlp_fusion.py` |
| Member 2 | CNN–LSTM fusion | `cnn_lstm.py` |
| Member 3 | VisualBERT | `visualbert.py` |
| Member 4 | CLIP classifier | `clip_classifier.py` |

Shared files (`common.py`, `README.md`, configs) are co-owned — whoever sets up the repo
commits them first.

### Set your identity before committing (do this once per member, per machine)

```bash
git config user.name  "Your Name"
git config user.email "your-regno@learner.manipal.edu"
```

### Suggested first commits (so the graph shows four distinct contributors)

```bash
# whoever bootstraps the repo
git add README.md common.py CONTRIBUTING.md .gitignore requirements.txt
git commit -m "Project scaffold: shared data split + metric protocol"

# Member 4
git add clip_classifier.py results/clip_results.json
git commit -m "CLIP classifier: frozen embeddings + MLP head; test_seen ROC-AUC 0.778"

# Member 1
git add mlp_fusion.py
git commit -m "MLP feature fusion: BERT[CLS] + ResNet-50 features"

# Member 2
git add cnn_lstm.py
git commit -m "CNN-LSTM fusion: ResNet-18 image branch + BiLSTM text branch"

# Member 3
git add visualbert.py
git commit -m "VisualBERT over torchvision region features (no detectron2)"
```

Keep committing as you iterate (feature extraction, tuning, bug fixes) — small frequent
commits per member are exactly the evidence the viva looks for.

### Verify the contribution split

```bash
git shortlog -sne        # commits per author
```
