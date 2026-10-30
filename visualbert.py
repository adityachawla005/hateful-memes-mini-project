"""
Model 3 — VisualBERT (single-stream vision-language transformer).   Owner: Aditya Chawla (Member 3).

Classic VisualBERT consumes pre-extracted object-region features. Instead of the brittle
detectron2 / MMF stack, we extract region features with torchvision's
`fasterrcnn_resnet50_fpn` (ROI-pooled box features), cache them once, and feed them to
Hugging Face `VisualBertModel`. A small trainable projection maps the torchvision region
dim (256, from the FPN feature map) to the 2048-d visual embedding VisualBERT expects.

Same split + same metric function as every other model (see common.py).

    Stage 1:  python visualbert.py --extract      # cache region features (slow, one-time)
    Stage 2:  python visualbert.py                # train + evaluate

----------------------------------------------------------------------------------------
FALLBACK (documented risk mitigation): if region extraction is too slow on the available
GPU budget, swap this slot for ViLT — a turnkey single-stream VL transformer that takes
raw pixels and needs no region features — while keeping the same comparison protocol:
    from transformers import ViltProcessor, ViltForImagesAndTextClassification
See the report's "Risk / Plan for Remaining Work" section.
----------------------------------------------------------------------------------------
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from transformers import BertTokenizerFast, VisualBertModel

from common import (
    BASE, WORK_DIR, TRAIN_SPLIT, EVAL_SPLITS, VAL_SPLIT, TEST_SPLIT,
    load_split, image_path, compute_metrics, save_results, set_seed,
)

REGION_DIR = os.path.join(WORK_DIR, "vb_regions")
CHECKPOINT_PATH = os.path.join(WORK_DIR, "visualbert.pt")
HF_NAME = "uclanlp/visualbert-vqa-coco-pre"
BERT_TOK = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_BOXES = 36
SCORE_THRESH = 0.2
MAX_TEXT_LEN = 64
VB_VISUAL_DIM = 2048   # what the pretrained VisualBERT expects

BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 10
PATIENCE = 3

os.makedirs(REGION_DIR, exist_ok=True)
set_seed()


# --------------------------------------------------------------------------- #
# Stage 1: region feature extraction (torchvision Faster R-CNN, no detectron2)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_regions(split):
    cache = os.path.join(REGION_DIR, f"{split}.npz")
    if os.path.exists(cache):
        return cache

    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
    from torchvision.ops import roi_align

    det = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE).eval()
    to_tensor = T.ToTensor()
    df = load_split(split).reset_index(drop=True)

    feats, masks, labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"regions[{split}]"):
        x = to_tensor(Image.open(image_path(row["img"])).convert("RGB")).to(DEVICE)
        pred = det([x])[0]
        boxes, scores = pred["boxes"], pred["scores"]
        keep = scores >= SCORE_THRESH
        boxes = boxes[keep][:MAX_BOXES] if keep.any() else boxes[:1]
        n = boxes.shape[0]
        fmap = det.backbone(x.unsqueeze(0))["0"]          # [1, 256, H, W]
        scale = fmap.shape[-1] / x.shape[-1]
        rois = torch.cat([torch.zeros(n, 1, device=DEVICE), boxes], dim=1)
        pooled = roi_align(fmap, rois, (1, 1), spatial_scale=scale,
                           aligned=True).flatten(1)        # [n, 256]
        C = pooled.shape[1]
        f = torch.zeros(MAX_BOXES, C); m = torch.zeros(MAX_BOXES)
        k = min(n, MAX_BOXES)
        f[:k] = pooled[:k].cpu(); m[:k] = 1.0
        feats.append(f.numpy()); masks.append(m.numpy())
        lab = int(row["label"]) if "label" in row and row["label"] == row["label"] else -1
        labels.append(lab)
    np.savez(cache, feat=np.stack(feats), mask=np.stack(masks),
             label=np.array(labels))
    print(f"saved {cache}")
    return cache


# --------------------------------------------------------------------------- #
# Stage 2: VisualBERT over cached regions
# --------------------------------------------------------------------------- #
class VBDataset(Dataset):
    def __init__(self, split, tokenizer):
        df = load_split(split).reset_index(drop=True)
        data = np.load(os.path.join(REGION_DIR, f"{split}.npz"))
        self.feat, self.mask, self.label = data["feat"], data["mask"], data["label"]
        enc = tokenizer(list(df["text"].astype(str)), padding="max_length",
                        truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.token_type = enc["token_type_ids"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attn[i],
            "token_type_ids": self.token_type[i],
            "visual_feat": torch.tensor(self.feat[i], dtype=torch.float32),
            "visual_mask": torch.tensor(self.mask[i], dtype=torch.float32),
            "label": torch.tensor(float(self.label[i])),
        }


class VisualBertClassifier(nn.Module):
    def __init__(self, region_dim):
        super().__init__()
        self.vbert = VisualBertModel.from_pretrained(HF_NAME)
        self.visual_proj = nn.Linear(region_dim, VB_VISUAL_DIM)
        hidden = self.vbert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids,
                visual_feat, visual_mask):
        visual_embeds = self.visual_proj(visual_feat)          # [B, boxes, 2048]
        vtt = torch.ones(visual_embeds.shape[:-1], dtype=torch.long,
                         device=visual_embeds.device)
        out = self.vbert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_mask,
            visual_token_type_ids=vtt,
        )
        return self.classifier(self.dropout(out.pooler_output)).squeeze(-1)


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, ys, ps = 0.0, [], []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        y = b.pop("label")
        with torch.set_grad_enabled(train):
            logit = model(b["input_ids"], b["attention_mask"], b["token_type_ids"],
                          b["visual_feat"], b["visual_mask"])
            loss = crit(logit, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * y.size(0)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    return tot / len(loader.dataset), np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true",
                    help="only run Stage 1 region extraction, then exit")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    for split in [TRAIN_SPLIT] + EVAL_SPLITS:
        extract_regions(split)
    if args.extract:
        print("region extraction complete."); return

    tok = BertTokenizerFast.from_pretrained(BERT_TOK)
    region_dim = np.load(os.path.join(REGION_DIR, f"{TRAIN_SPLIT}.npz"))["feat"].shape[-1]
    loaders = {s: DataLoader(VBDataset(s, tok), batch_size=BATCH_SIZE,
                             shuffle=(s == TRAIN_SPLIT), num_workers=2)
               for s in [TRAIN_SPLIT] + EVAL_SPLITS}

    model = VisualBertClassifier(region_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
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
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nBest val AUC: {best_auc:.4f}")

    print("\n--- final evaluation ---")
    results = []
    for split in EVAL_SPLITS:
        _, y, p = run_epoch(model, loaders[split], crit)
        results.append(compute_metrics(y, p, split))
    save_results("visualbert", results)


if __name__ == "__main__":
    main()
