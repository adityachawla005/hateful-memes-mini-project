"""
Model 4 — CLIP-based Classifier.   Owner: Sahithya Sambhrama (Member 4).   BEST MODEL.

Freeze a pretrained CLIP model, extract image+text embeddings once per split (cached),
then train a small MLP head on the concatenated L2-normalized embeddings.

Final configuration uses CLIP ViT-L/14 (test_seen ROC-AUC 0.822). Set CLIP_MODEL_NAME to
"openai/clip-vit-base-patch32" to reproduce the ViT-B/32 baseline (0.777). An ablation with
a cross-modal interaction head on ViT-B/32 gave no gain, indicating the backbone — not the
fusion — is the bottleneck.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import (WORK_DIR, TRAIN_SPLIT, EVAL_SPLITS, VAL_SPLIT,
                    load_split, image_path, compute_metrics, save_results, set_seed)

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"     # baseline: openai/clip-vit-base-patch32
FEATURE_DIR = os.path.join(WORK_DIR, "clip_features_large")
CHECKPOINT_PATH = os.path.join(WORK_DIR, "clip_large_head.pt")
RESULT_NAME = "clip_large" if "large" in CLIP_MODEL_NAME else "clip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 128
HIDDEN_DIM = 512
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 3e-4
EPOCHS = 30
PATIENCE = 6
os.makedirs(FEATURE_DIR, exist_ok=True)
set_seed()


class MemeRawDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(image_path(row["img"])).convert("RGB")
        label = int(row["label"]) if "label" in row and row["label"] == row["label"] else -1
        return image, row["text"], label


def collate_raw(batch):
    images, texts, labels = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long)


def extract_features(split, clip_model, processor):
    cache = os.path.join(FEATURE_DIR, f"{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {"image": d["image"], "text": d["text"], "label": d["label"]}
    loader = DataLoader(MemeRawDataset(load_split(split)), batch_size=FEATURE_BATCH_SIZE,
                        shuffle=False, collate_fn=collate_raw, num_workers=2)
    img_f, txt_f, labs = [], [], []
    clip_model.eval()
    with torch.no_grad():
        for images, texts, batch_labels in tqdm(loader, desc=f"extract[{split}]"):
            inp = processor(text=texts, images=images, return_tensors="pt",
                            padding=True, truncation=True).to(DEVICE)
            out = clip_model(pixel_values=inp["pixel_values"], input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"])
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            img_f.append(ie.cpu().numpy()); txt_f.append(te.cpu().numpy())
            labs.append(batch_labels.numpy())
    feats = {"image": np.concatenate(img_f), "text": np.concatenate(txt_f),
             "label": np.concatenate(labs)}
    np.savez(cache, **feats)
    return feats


def feat_tensor(feats):
    return torch.tensor(np.concatenate([feats["image"], feats["text"]], axis=1),
                        dtype=torch.float32)


class ClipHead(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(train_feats, val_feats):
    Xtr = feat_tensor(train_feats); ytr = torch.tensor(train_feats["label"], dtype=torch.float32)
    Xval = feat_tensor(val_feats).to(DEVICE); yval = val_feats["label"]
    model = ClipHead(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            vp = 1 / (1 + np.exp(-model(Xval).cpu().numpy()))
        vauc = roc_auc_score(yval, vp)
        print(f"epoch {epoch:02d}  train_loss={tot/len(Xtr):.4f}  val_auc={vauc:.4f}")
        if vauc > best_auc:
            best_auc, bad = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {epoch}"); break
    model.load_state_dict(best_state); torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nBest val AUC: {best_auc:.4f}")
    return model


def evaluate(model, feats, split_name):
    X = feat_tensor(feats).to(DEVICE); model.eval()
    with torch.no_grad():
        probs = 1 / (1 + np.exp(-model(X).cpu().numpy()))
    return compute_metrics(feats["label"], probs, split_name)


def main():
    print(f"device: {DEVICE}\nloading CLIP: {CLIP_MODEL_NAME}")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    for p in clip_model.parameters():
        p.requires_grad = False
    print("\n--- extracting / loading cached features ---")
    train_feats = extract_features(TRAIN_SPLIT, clip_model, processor)
    eval_feats = {s: extract_features(s, clip_model, processor) for s in EVAL_SPLITS}
    print("\n--- training classifier head (dev_seen used for model selection) ---")
    model = train_head(train_feats, eval_feats[VAL_SPLIT])
    print("\n--- final evaluation ---")
    results = [evaluate(model, eval_feats[s], s) for s in EVAL_SPLITS]
    save_results(RESULT_NAME, results)


if __name__ == "__main__":
    main()
