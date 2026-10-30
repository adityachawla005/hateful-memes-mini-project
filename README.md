# Multimodal Hate Speech Detection in Memes — Comparative Deep Learning Architectures

**ICT-4442 Deep Learning Mini Project.** Binary hateful / non-hateful meme classification on the
**Hateful Memes** dataset (Facebook AI, NeurIPS 2020), comparing four architectures from different
families under one common protocol.

| Model | Family | Owner | Script | test_seen ROC-AUC |
|---|---|---|---|---|
| CLIP classifier (ViT-L/14) | Contrastive vision–language embeddings | Sahithya Sambhrama (230911014) | `clip_classifier.py` | ✅ **0.822** (best; ViT-B/32 baseline 0.777) |
| MLP feature fusion | Classical feature fusion | Parth Khator (230911422) | `mlp_fusion.py` | ✅ **0.700** |
| CNN–LSTM fusion | CNN + RNN fusion | Sarthak Atrish (230911504) | `cnn_lstm.py` | ✅ **0.682** |
| VisualBERT | Single-stream VL transformer | Aditya Chawla (230911260) | `visualbert.py` | ⚙️ in progress (region extraction) |

Final comparison and per-model metrics: `results/results.md`. Improvement ablation: CLIP
ViT-B/32→L/14 (0.777→0.822); MLP +mean-pool/interaction (0.677→0.700); CNN–LSTM +GloVe/frozen
backbone (0.600→0.682); cross-modal head on CLIP-B/32 gave no gain — backbone, not fusion, was
the bottleneck.

Primary metric: **ROC-AUC** (benchmark-consistent). Supplementary: accuracy, precision, recall, F1.

> **Academic integrity.** Every number in this repo is produced by actually running the code.
> `results/clip_results.json` holds the real CLIP run; the other models write their own JSON when
> you run them. Nothing is hard-coded or invented.

---

## Split policy (identical for all four models)

```
train.jsonl      -> training
dev_seen.jsonl   -> validation / model selection (early stopping)
test_seen.jsonl  -> final reported metrics
```

The official `test_unseen` labels are withheld, so we do not score on them. Keeping one split and
one metric function (`common.compute_metrics`) across all models is what makes the comparison fair.

## Dataset

Access-gated (Meta data licence) — **not** included in this repo. Attach the Kaggle mirror
`williamberrios/hateful-memes` to a Kaggle notebook, or set `HM_BASE` to your local copy:

```bash
export HM_BASE=/path/to/hateful_memes      # folder containing img/ and the *.jsonl files
export HM_WORK=./_work                      # where caches/results are written (default /kaggle/working)
```

## Running

```bash
pip install -r requirements.txt            # Kaggle already has most of these

python clip_classifier.py                  # Member 4  (reproduces ROC-AUC ~0.778)
python mlp_fusion.py                        # Member 1
python cnn_lstm.py                          # Member 2
python visualbert.py --extract             # Member 3, stage 1: cache region features (slow)
python visualbert.py                        # Member 3, stage 2: train + evaluate

python make_results_table.py               # -> results/results.csv + results/results.md
```

Each model script prints `[dev_seen]` and `[test_seen]` metric lines and writes
`results/<model>_results.json`. `make_results_table.py` collates them into the single comparison
table used in the report.

## Layout

```
common.py             # shared: data split, image paths, the ONE metric function, result I/O
clip_classifier.py    # Member 4
mlp_fusion.py         # Member 1
cnn_lstm.py           # Member 2
visualbert.py         # Member 3  (+ torchvision region extraction, no detectron2)
make_results_table.py # collate results -> comparison table
results/              # per-model metric JSONs (clip_results.json is real)
CONTRIBUTING.md       # per-member git ownership so commit history evidences contribution
```

## Notes on what to expect

The benchmark is adversarial (confounders defeat single-modality shortcuts), so absolute numbers
are modest by design. Published ballpark: simple fusion ≈ 0.63–0.66 AUROC, VisualBERT-class
≈ 0.71–0.75, humans ≈ 0.85. Our CLIP head already reaches **0.778** on `test_seen`, which is a
strong result for this task. The graded contribution is the *comparison and analysis*, not chasing
state of the art.

**VisualBERT risk & fallback:** classic VisualBERT needs region features. We extract them with
torchvision Faster R-CNN (cached) to avoid the brittle detectron2/MMF setup. If that is too slow on
the GPU budget, `visualbert.py` documents swapping in **ViLT** (raw-pixel single-stream VL
transformer, no region features) while keeping the same comparison slot and protocol.
