"""
Model 1 — MLP Feature Fusion.   Owner: Parth Khator (Member 1).   test_seen ROC-AUC 0.700.

Frozen BERT text features (masked MEAN-pool over tokens) ⊕ frozen ResNet-50 image features,
each projected to a common space and fused as [img, txt, img*txt] before an MLP classifier.
Backbones are frozen and cached once.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common import (WORK_DIR, TRAIN_SPLIT, EVAL_SPLITS, VAL_SPLIT,
                    load_split, image_path, compute_metrics, save_results, set_seed)

FEATURE_DIR = os.path.join(WORK_DIR, "mlp_features_v2")
CHECKPOINT_PATH = os.path.join(WORK_DIR, "mlp_fusion_v2.pt")
BERT_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TEXT_LEN = 64
FEATURE_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 128
PROJ_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 40
PATIENCE = 8
os.makedirs(FEATURE_DIR, exist_ok=True)
set_seed()

_img_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def masked_mean(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def extract_features(split, bert, tok, resnet):
    cache = os.path.join(FEATURE_DIR, f"{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {"image": d["image"], "text": d["text"], "label": d["label"]}
    df = load_split(split).reset_index(drop=True)
    img_f, txt_f, labs = [], [], []
    bert.eval(); resnet.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(df), FEATURE_BATCH_SIZE), desc=f"extract[{split}]"):
            rows = df.iloc[i:i + FEATURE_BATCH_SIZE]
            enc = tok(list(rows["text"].astype(str)), padding=True, truncation=True,
                      max_length=MAX_TEXT_LEN, return_tensors="pt").to(DEVICE)
            txt = masked_mean(bert(**enc).last_hidden_state, enc["attention_mask"])
            imgs = torch.stack([
                _img_tf(Image.open(image_path(p)).convert("RGB")) for p in rows["img"]
            ]).to(DEVICE)
            im = resnet(imgs)
            img_f.append(im.cpu().numpy()); txt_f.append(txt.cpu().numpy())
            labs.append(np.array([int(x) if x == x else -1 for x in rows["label"]]))
    feats = {"image": np.concatenate(img_f), "text": np.concatenate(txt_f),
             "label": np.concatenate(labs)}
    np.savez(cache, **feats)
    return feats


class GatedFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, proj=PROJ_DIM, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.img_proj = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, proj), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.LayerNorm(txt_dim), nn.Linear(txt_dim, proj), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(proj * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, img, txt):
        i = self.img_proj(img); t = self.txt_proj(txt)
        return self.net(torch.cat([i, t, i * t], dim=-1)).squeeze(-1)


def to_tensors(feats):
    return (torch.tensor(feats["image"], dtype=torch.float32),
            torch.tensor(feats["text"], dtype=torch.float32),
            torch.tensor(feats["label"], dtype=torch.float32))


def train_model(train_feats, val_feats):
    itr, ttr, ytr = to_tensors(train_feats)
    iv, tv, _ = to_tensors(val_feats); iv, tv = iv.to(DEVICE), tv.to(DEVICE)
    yval = val_feats["label"]
    model = GatedFusion(itr.shape[1], ttr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(itr, ttr, ytr), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for ib, tb, yb in loader:
            ib, tb, yb = ib.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(ib, tb), yb); loss.backward(); opt.step()
            tot += loss.item() * ib.size(0)
        model.eval()
        with torch.no_grad():
            vp = 1 / (1 + np.exp(-model(iv, tv).cpu().numpy()))
        vauc = roc_auc_score(yval, vp)
        print(f"epoch {epoch:02d}  train_loss={tot/len(itr):.4f}  val_auc={vauc:.4f}")
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
    i, t, _ = to_tensors(feats); i, t = i.to(DEVICE), t.to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = 1 / (1 + np.exp(-model(i, t).cpu().numpy()))
    return compute_metrics(feats["label"], probs, split_name)


def main():
    print(f"device: {DEVICE}\nloading BERT ({BERT_NAME}) + ResNet-50 ...")
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    bert = AutoModel.from_pretrained(BERT_NAME).to(DEVICE)
    resnet = models.resnet50(weights="IMAGENET1K_V2"); resnet.fc = nn.Identity()
    resnet = resnet.to(DEVICE)
    for p in list(bert.parameters()) + list(resnet.parameters()):
        p.requires_grad = False
    print("\n--- extracting / loading cached features ---")
    train_feats = extract_features(TRAIN_SPLIT, bert, tok, resnet)
    eval_feats = {s: extract_features(s, bert, tok, resnet) for s in EVAL_SPLITS}
    print("\n--- training fusion model (dev_seen used for model selection) ---")
    model = train_model(train_feats, eval_feats[VAL_SPLIT])
    print("\n--- final evaluation ---")
    results = [evaluate(model, eval_feats[s], s) for s in EVAL_SPLITS]
    save_results("mlp_fusion", results)


if __name__ == "__main__":
    main()
