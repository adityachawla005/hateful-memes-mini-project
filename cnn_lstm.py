"""
Model 2 — CNN–LSTM Fusion.   Owner: Member 2.   test_seen ROC-AUC 0.682.

Two-branch model: a FROZEN ResNet-18 image extractor + a BiLSTM text branch initialised with
pretrained GloVe-200d embeddings. Freezing the CNN and using GloVe removes the overfitting that
sank the from-scratch baseline (val AUC used to peak at epoch 3 then collapse).
"""
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import roc_auc_score

from common import (WORK_DIR, TRAIN_SPLIT, EVAL_SPLITS, VAL_SPLIT, TEST_SPLIT,
                    load_split, image_path, compute_metrics, save_results, set_seed)

CHECKPOINT_PATH = os.path.join(WORK_DIR, "cnn_lstm_v2.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TEXT_LEN = 64
EMBED_DIM = 200
LSTM_HIDDEN = 256
FUSION_DIM = 256
DROPOUT = 0.4
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 25
PATIENCE = 6
MIN_FREQ = 2
set_seed()

_tok_re = re.compile(r"[a-z0-9']+")
PAD, UNK = "<pad>", "<unk>"


def tokenize(text):
    return _tok_re.findall(str(text).lower())


def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    itos = [PAD, UNK] + [w for w, n in c.items() if n >= MIN_FREQ]
    return {w: i for i, w in enumerate(itos)}, itos


def build_embeddings(itos):
    emb = np.random.normal(scale=0.1, size=(len(itos), EMBED_DIM)).astype(np.float32)
    emb[0] = 0.0
    try:
        import gensim.downloader as api
        print("loading GloVe (glove-wiki-gigaword-200) via gensim ...")
        glove = api.load("glove-wiki-gigaword-200")
        hit = 0
        for i, w in enumerate(itos):
            if w in glove:
                emb[i] = glove[w]; hit += 1
        print(f"GloVe initialised {hit}/{len(itos)} tokens ({100*hit/len(itos):.1f}%)")
    except Exception as e:
        print(f"[warn] GloVe unavailable ({e}); using random embeddings.")
    return torch.tensor(emb)


def encode_text(text, stoi):
    ids = [stoi.get(w, 1) for w in tokenize(text)][:MAX_TEXT_LEN]
    length = max(len(ids), 1)
    if len(ids) < MAX_TEXT_LEN:
        ids = ids + [0] * (MAX_TEXT_LEN - len(ids))
    return ids, length


_train_tf = T.Compose([
    T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_eval_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MemeDataset(Dataset):
    def __init__(self, df, stoi, train=False):
        self.df = df.reset_index(drop=True); self.stoi = stoi
        self.tf = _train_tf if train else _eval_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.tf(Image.open(image_path(row["img"])).convert("RGB"))
        ids, length = encode_text(row["text"], self.stoi)
        label = float(row["label"]) if "label" in row and row["label"] == row["label"] else 0.0
        return (img, torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long), torch.tensor(label))


class CNNLSTM(nn.Module):
    def __init__(self, emb_weights):
        super().__init__()
        cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.img_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()
        for p in cnn.parameters():
            p.requires_grad = False     # frozen image extractor
        self.cnn = cnn
        self.embed = nn.Embedding.from_pretrained(emb_weights, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.img_proj = nn.Sequential(nn.Linear(self.img_dim, FUSION_DIM), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.Linear(2 * LSTM_HIDDEN, FUSION_DIM), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(2 * FUSION_DIM, FUSION_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM, 1))

    def forward(self, img, ids, lengths):
        with torch.no_grad():
            feat = self.cnn(img)
        im = self.img_proj(feat)
        emb = self.embed(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        txt = self.txt_proj(torch.cat([h[-2], h[-1]], dim=-1))
        return self.head(torch.cat([im, txt], dim=-1)).squeeze(-1)


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, ys, ps = 0.0, [], []
    for img, ids, length, y in loader:
        img, ids, length, y = img.to(DEVICE), ids.to(DEVICE), length.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logit = model(img, ids, length)
            loss = crit(logit, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * y.size(0)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    return tot / len(loader.dataset), np.concatenate(ys), np.concatenate(ps)


def main():
    print(f"device: {DEVICE}")
    dfs = {s: load_split(s) for s in [TRAIN_SPLIT] + EVAL_SPLITS}
    stoi, itos = build_vocab(dfs[TRAIN_SPLIT]["text"])
    print(f"vocab size: {len(stoi)}")
    emb_weights = build_embeddings(itos)
    loaders = {
        TRAIN_SPLIT: DataLoader(MemeDataset(dfs[TRAIN_SPLIT], stoi, train=True),
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        VAL_SPLIT: DataLoader(MemeDataset(dfs[VAL_SPLIT], stoi), batch_size=BATCH_SIZE, num_workers=2),
        TEST_SPLIT: DataLoader(MemeDataset(dfs[TEST_SPLIT], stoi), batch_size=BATCH_SIZE, num_workers=2),
    }
    model = CNNLSTM(emb_weights).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, _, _ = run_epoch(model, loaders[TRAIN_SPLIT], crit, opt)
        _, yval, pval = run_epoch(model, loaders[VAL_SPLIT], crit)
        vauc = roc_auc_score(yval, pval)
        print(f"epoch {epoch:02d}  train_loss={tr_loss:.4f}  val_auc={vauc:.4f}")
        if vauc > best_auc:
            best_auc, bad = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {epoch}"); break
    model.load_state_dict(best_state); torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nBest val AUC: {best_auc:.4f}\n\n--- final evaluation ---")
    results = []
    for split in EVAL_SPLITS:
        _, y, p = run_epoch(model, loaders[split], crit)
        results.append(compute_metrics(y, p, split))
    save_results("cnn_lstm", results)


if __name__ == "__main__":
    main()
