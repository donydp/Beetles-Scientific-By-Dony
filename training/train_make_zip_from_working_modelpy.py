#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ziptrue_final.py
- Train event-level model for SMOOD Sentinel Beetles
- Outputs submission.zip ready for Codabench:
  - model.py
  - model_weights.pth
  - encoders.pkl
  - lnt_calibration.npz
  - requirements.txt (empty)
  - requirements.txt.txt (empty)
Fixes / Improvements:
- Robust HF image column detection: prefers actual decoded images, avoids string path columns
- Stable SAM + AMP: use autocast, but disable GradScaler when SAM is enabled (prevents unscale_ stage errors)
- No scaler.update() without step (prevents "No inf checks were recorded" assertion)
- Long-horizon (1y, 2y) uncertainty calibration stronger to improve OOD 1y/2y CRPS
"""

import os
import io
import math
import pickle
import random
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import mobilenet_v3_small

try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("Please install datasets: pip install datasets") from e


# ============================================================
# CONFIG
# ============================================================
SEED = 42
HF_DATASET_NAME = "imageomics/sentinel-beetles"
HF_TOKEN = None  # or set env var / token if needed

IMAGE_SIDE = 160
MAX_IMAGES_PER_EVENT = 4

BATCH_SIZE = 16
EPOCHS = 60

LR = 2e-4
WEIGHT_DECAY = 3e-4
GRAD_CLIP_NORM = 1.0

UNFREEZE_BACKBONE_AT_EPOCH = 10
BACKBONE_LR_MULT = 0.2

USE_SAM = True
SAM_RHO = 0.05

USE_AMP = True  # autocast enabled on cuda
# IMPORTANT:
# - If USE_SAM=True, GradScaler is disabled to avoid optimizer-stage unscale errors.
# - If USE_SAM=False, GradScaler enabled normally.

SAAB_K_DIM = 192
SAAB_ENERGY_THRESHOLD = 0.997
SAAB_MIN_EIG_RATIO = 1e-5
SAAB_FIT_MAX_SAMPLES = 60000
SAAB_FIT_MAX_EVENTS = 16000

META_EMBED_DIM = 96
META_DROPOUT = 0.10
META_DROP_PROB_TRAIN = 0.10

USE_PRIORS = True
PRI_DIM = 12

# Loss weights
LAMBDA_RMSE = 0.10
LAMBDA_SIGMA_REG = 0.001

# EMA
USE_EMA = True
EMA_DECAY = 0.999

# split
ID_VAL_FRAC_PER_DOMAIN = 0.10

NUM_WORKERS = 0
PIN_MEMORY = True

# Calibration
USE_VARIANCE_TEMP = True
VT_STEPS = 500
VT_LR = 0.04

# Extra: long-horizon sigma floor (helps OOD CRPS by avoiding overconfident long forecasts)
SIGMA_FLOOR_LONG_STD_FRAC = 0.06  # fraction of target std for 1y & 2y only


# ============================================================
# PATHS
# ============================================================
ROOT = Path.cwd()
OUT_DIR = ROOT / "artifacts_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_FILENAME = "model_weights.pth"
ENCODERS_FILENAME = "encoders.pkl"
LNT_FILENAME = "lnt_calibration.npz"

WEIGHTS_PATH = OUT_DIR / WEIGHT_FILENAME
ENCODERS_PATH = OUT_DIR / ENCODERS_FILENAME
LNT_PATH = OUT_DIR / LNT_FILENAME

ZIP_PATH = ROOT / "submission.zip"


# ============================================================
# REPRO & DEVICE
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

amp_enabled = bool(USE_AMP and device.type == "cuda")
# Disable GradScaler if SAM is used (prevents "already unscaled" stage mismatch)
scaler_enabled = bool(amp_enabled and (not USE_SAM))
scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)


# ============================================================
# HF LOAD
# ============================================================
def safe_load_hf_dataset(name: str):
    print("Loading HuggingFace dataset:", name)
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return load_dataset(name, **kwargs)


# ============================================================
# IMAGE HELPERS
# ============================================================
def _open_image_any(maybe_img: Any, image_side: int) -> Image.Image:
    def black():
        return Image.new("RGB", (image_side, image_side), (0, 0, 0))

    if maybe_img is None:
        return black()

    try:
        if isinstance(maybe_img, Image.Image):
            return maybe_img.convert("RGB")
        if isinstance(maybe_img, (bytes, bytearray)):
            return Image.open(io.BytesIO(maybe_img)).convert("RGB")
        if isinstance(maybe_img, dict):
            if maybe_img.get("bytes") is not None:
                return Image.open(io.BytesIO(maybe_img["bytes"])).convert("RGB")
            p = maybe_img.get("path")
            if isinstance(p, str) and os.path.exists(p):
                return Image.open(p).convert("RGB")
            return black()
        if isinstance(maybe_img, str):
            if os.path.exists(maybe_img):
                return Image.open(maybe_img).convert("RGB")
            return black()
    except Exception:
        return black()

    return black()


def make_transforms(image_side: int):
    tr = transforms.Compose([
        transforms.RandomResizedCrop(image_side, scale=(0.75, 1.0), ratio=(0.90, 1.12)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(0.12, 0.12, 0.12, 0.04),
        transforms.RandomAutocontrast(p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tr = transforms.Compose([
        transforms.Resize((image_side, image_side)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tr, val_tr


def detect_image_column(hf_dataset) -> Optional[str]:
    """
    IMPORTANT:
    - Prefer columns that decode to PIL.Image or dict{bytes}.
    - Avoid path-only columns like file_path / relative_img_loc unless no other choice.
    """
    cols = list(getattr(hf_dataset, "column_names", []))
    if not cols:
        return None

    # 1) strongly preferred names that usually are actual decoded images
    preferred = ["relative_img", "image", "beetle_img", "specimen_img", "img"]
    for c in preferred:
        if c in cols:
            # verify quickly
            try:
                v = hf_dataset[0].get(c)
                if isinstance(v, Image.Image):
                    return c
                if isinstance(v, dict) and v.get("bytes") is not None:
                    return c
            except Exception:
                pass

    # 2) scan for actual image-like columns
    sample_n = min(64, len(hf_dataset))
    best = None
    best_ok = -1
    for c in cols:
        ok = 0
        for i in range(sample_n):
            try:
                v = hf_dataset[i].get(c)
            except Exception:
                v = None
            if isinstance(v, Image.Image):
                ok += 1
            elif isinstance(v, (bytes, bytearray)):
                ok += 1
            elif isinstance(v, dict) and (v.get("bytes") is not None):
                ok += 1
        if ok > best_ok:
            best_ok = ok
            best = c

    if best is not None and best_ok >= max(2, sample_n // 8):
        return best

    # 3) last resort: allow path columns (will likely hurt training)
    fallback = ["relative_img_loc", "file_path", "filepath", "path", "image_path"]
    for c in fallback:
        if c in cols:
            return c

    return None


# ============================================================
# EVENT INDEX BUILDER
# ============================================================
def build_event_index_hf(hf_dataset) -> List[Dict[str, Any]]:
    if hf_dataset is None:
        return []
    n = len(hf_dataset)
    cols = set(getattr(hf_dataset, "column_names", []))

    def col(name: str):
        if name in cols:
            return hf_dataset[name]
        return [None] * n

    img_col = detect_image_column(hf_dataset)
    if img_col is None:
        print("[WARN] Could not detect image column. Columns:", sorted(list(cols))[:80])
        return []

    eventID = col("eventID")
    scientificName = col("scientificName")
    domainID = col("domainID")
    siteID = col("siteID")

    t30 = col("SPEI_30d")
    t1y = col("SPEI_1y")
    t2y = col("SPEI_2y")

    events = defaultdict(lambda: {
        "row_indices": [],
        "scientificName": None,
        "domainID": None,
        "siteID": None,
        "targets": {},
        "img_col": img_col,
    })

    for i in range(n):
        eid = eventID[i] if eventID[i] is not None else f"event_{i}"
        e = events[eid]
        e["row_indices"].append(i)

        if e["scientificName"] is None:
            e["scientificName"] = scientificName[i]
        if e["domainID"] is None:
            e["domainID"] = domainID[i]
        if e["siteID"] is None:
            e["siteID"] = siteID[i]

        if t30[i] is not None:
            e["targets"]["SPEI_30d"] = float(t30[i])
        if t1y[i] is not None:
            e["targets"]["SPEI_1y"] = float(t1y[i])
        if t2y[i] is not None:
            e["targets"]["SPEI_2y"] = float(t2y[i])

    out = []
    for eid, v in events.items():
        if len(v["row_indices"]) == 0:
            continue
        if len(v["targets"]) != 3:
            continue
        out.append({
            "eventID": eid,
            "scientificName": v["scientificName"],
            "domainID": v["domainID"],
            "siteID": v["siteID"],
            "row_indices": v["row_indices"],
            "targets": v["targets"],
            "img_col": v["img_col"],
        })
    return out


# ============================================================
# ENCODERS & PRIORS
# ============================================================
def make_mapping(values: List[Any]) -> Dict[str, int]:
    uniq = sorted({str(v) for v in values if v is not None and str(v) != ""})
    return {k: i + 1 for i, k in enumerate(uniq)}  # 0 reserved for UNK


def build_encoders(train_events: List[Dict[str, Any]], extra_events: Optional[List[Dict[str, Any]]] = None):
    all_events = train_events + (extra_events or [])
    species_map = make_mapping([e.get("scientificName") for e in all_events])
    domain_map = make_mapping([e.get("domainID") for e in all_events])
    site_map = make_mapping([e.get("siteID") for e in all_events])
    return species_map, domain_map, site_map


def compute_target_stats(events: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    ys = []
    for e in events:
        t = e.get("targets")
        if not t:
            continue
        ys.append([t["SPEI_30d"], t["SPEI_1y"], t["SPEI_2y"]])
    y = np.asarray(ys, dtype=np.float32)
    mu = y.mean(axis=0)
    sd = np.maximum(y.std(axis=0), 1e-3)
    return mu, sd


def build_priors(events: List[Dict[str, Any]], y_mean: np.ndarray, y_std: np.ndarray):
    def upd(store, key, y_s):
        if key is None:
            return
        k = str(key)
        if k not in store:
            store[k] = {"sum": np.zeros((3,), np.float32), "cnt": 0}
        store[k]["sum"] += y_s
        store[k]["cnt"] += 1

    sp, dm, st = {}, {}, {}
    for e in events:
        t = e["targets"]
        y = np.array([t["SPEI_30d"], t["SPEI_1y"], t["SPEI_2y"]], np.float32)
        y_s = (y - y_mean) / y_std
        upd(sp, e.get("scientificName"), y_s)
        upd(dm, e.get("domainID"), y_s)
        upd(st, e.get("siteID"), y_s)

    def finalize(store):
        out_mean, out_cnt = {}, {}
        for k, v in store.items():
            cnt = max(1, int(v["cnt"]))
            out_mean[k] = (v["sum"] / float(cnt)).astype(np.float32)
            out_cnt[k] = int(cnt)
        return out_mean, out_cnt

    sp_mean, sp_cnt = finalize(sp)
    dm_mean, dm_cnt = finalize(dm)
    st_mean, st_cnt = finalize(st)
    return sp_mean, sp_cnt, dm_mean, dm_cnt, st_mean, st_cnt


# ============================================================
# DATASET
# ============================================================
class EventDataset(Dataset):
    def __init__(self, events, hf_dataset, transform, max_images, shuffle_images,
                 priors: Optional[Dict[str, Any]] = None):
        self.events = events
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.max_images = int(max_images)
        self.shuffle_images = bool(shuffle_images)
        self.priors = priors or {}

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        e = self.events[idx]
        indices = list(e["row_indices"])
        if self.shuffle_images:
            random.shuffle(indices)
        indices = indices[: self.max_images]

        imgs = []
        img_col = e.get("img_col")
        for ridx in indices:
            row = self.hf_dataset[int(ridx)]
            maybe_img = row.get(img_col) if img_col is not None else None
            im = _open_image_any(maybe_img, IMAGE_SIDE)
            imgs.append(self.transform(im))

        if len(imgs) == 0:
            imgs = [self.transform(Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), (0, 0, 0)))]

        images_t = torch.stack(imgs, dim=0)

        target = torch.tensor(
            [e["targets"]["SPEI_30d"], e["targets"]["SPEI_1y"], e["targets"]["SPEI_2y"]],
            dtype=torch.float32
        )

        pri = torch.zeros((PRI_DIM,), dtype=torch.float32)
        if USE_PRIORS and self.priors:
            p = self.priors
            sp_mean = p.get("sp_mean", {}) or {}
            dm_mean = p.get("dm_mean", {}) or {}
            st_mean = p.get("st_mean", {}) or {}
            sp_cnt = p.get("sp_cnt", {}) or {}
            dm_cnt = p.get("dm_cnt", {}) or {}
            st_cnt = p.get("st_cnt", {}) or {}

            sk = str(e.get("scientificName"))
            dk = str(e.get("domainID"))
            tk = str(e.get("siteID"))

            vsp = sp_mean.get(sk)
            vdm = dm_mean.get(dk)
            vst = st_mean.get(tk)

            if vsp is not None: pri[0:3] = torch.from_numpy(np.asarray(vsp, np.float32))
            if vdm is not None: pri[3:6] = torch.from_numpy(np.asarray(vdm, np.float32))
            if vst is not None: pri[6:9] = torch.from_numpy(np.asarray(vst, np.float32))

            pri[9]  = math.log1p(float(sp_cnt.get(sk, 0))) / 6.0
            pri[10] = math.log1p(float(dm_cnt.get(dk, 0))) / 6.0
            pri[11] = math.log1p(float(st_cnt.get(tk, 0))) / 6.0

        return {
            "eventID": e["eventID"],
            "images": images_t,
            "scientificName": e.get("scientificName"),
            "domainID": e.get("domainID"),
            "siteID": e.get("siteID"),
            "target": target,
            "pri": pri
        }


def collate_events(batch):
    return batch


# ============================================================
# MODEL
# ============================================================
class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = mobilenet_v3_small(weights=weights)
        except Exception:
            m = mobilenet_v3_small(pretrained=pretrained)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class FastEventModelMultiHead(nn.Module):
    def __init__(self, k_dim: int, proj_dim: int,
                 species_vocab: int, domain_vocab: int, site_vocab: int,
                 meta_embed_dim: int, meta_dropout: float,
                 pri_dim: int, max_images_per_event: int,
                 backbone_pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.max_images_per_event = int(max_images_per_event)

        self.backbone = MobileNetV3SmallBackbone(pretrained=backbone_pretrained)
        self.saab_linear = nn.Linear(576, int(k_dim), bias=True)

        self.project = nn.Sequential(
            nn.Linear(int(k_dim), int(proj_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.species_emb = nn.Embedding(max(1, int(species_vocab)), int(meta_embed_dim))
        self.domain_emb  = nn.Embedding(max(1, int(domain_vocab)),  int(meta_embed_dim))
        self.site_emb    = nn.Embedding(max(1, int(site_vocab)),    int(meta_embed_dim))
        self.meta_drop = nn.Dropout(float(meta_dropout))

        fused_dim = (2 * int(proj_dim)) + (3 * int(meta_embed_dim)) + 2 + int(pri_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 384),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        def head_block():
            return nn.Sequential(
                nn.Linear(192, 128),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(128, 2),
            )
        self.head_0 = head_block()
        self.head_1 = head_block()
        self.head_2 = head_block()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, images_list: List[torch.Tensor],
                sp_idx: torch.Tensor, dm_idx: torch.Tensor, st_idx: torch.Tensor,
                pri: torch.Tensor):
        dev = next(self.parameters()).device
        lengths = [int(x.shape[0]) for x in images_list]
        x = torch.cat([t.to(dev, non_blocking=True) for t in images_list], dim=0)

        feats = self.backbone(x).float()
        z = self.saab_linear(feats).float()
        z = self.project(z)

        splits = torch.split(z, lengths, dim=0)
        pooled_list = []
        for f in splits:
            mean = f.mean(dim=0)
            mx = f.max(dim=0).values
            pooled_list.append(torch.cat([mean, mx], dim=0))
        pooled = torch.stack(pooled_list, dim=0)

        sp = self.meta_drop(self.species_emb(sp_idx.to(dev)))
        dm = self.meta_drop(self.domain_emb(dm_idx.to(dev)))
        st = self.meta_drop(self.site_emb(st_idx.to(dev)))

        n_imgs = torch.tensor(lengths, dtype=torch.float32, device=dev).view(-1, 1)
        n_imgs_norm = n_imgs / float(self.max_images_per_event)
        n_imgs_log  = torch.log1p(n_imgs)

        pri = pri.to(dev, dtype=torch.float32)
        fused = torch.cat([pooled, sp, dm, st, n_imgs_norm, n_imgs_log, pri], dim=1)

        h = self.shared(fused)

        o0 = self.head_0(h)
        o1 = self.head_1(h)
        o2 = self.head_2(h)

        mu = torch.stack([o0[:, 0], o1[:, 0], o2[:, 0]], dim=1)
        log_sigma = torch.stack([o0[:, 1], o1[:, 1], o2[:, 1]], dim=1)
        log_sigma = torch.clamp(log_sigma, -7.0, 3.0)
        return mu, log_sigma


# ============================================================
# LOSS: Gaussian CRPS
# ============================================================
SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

def gaussian_crps_per_sample(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_sigma = torch.clamp(log_sigma, -7.0, 3.0)
    sigma = torch.exp(log_sigma).clamp_min(1e-6)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / SQRT_2PI
    Phi = 0.5 * (1.0 + torch.erf(z / SQRT_2))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - INV_SQRT_PI)
    return crps.mean(dim=1)

def rmse_per_sample(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((mu - y) ** 2, dim=1) + 1e-12)

def total_loss_per_sample(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Slightly emphasize long horizons (1y,2y) without destroying 30d
    # Weight inside CRPS by scaling targets
    w = torch.tensor([1.0, 1.10, 1.18], device=mu.device).view(1, 3)
    crps_i = gaussian_crps_per_sample(mu * w, log_sigma, y * w)
    rmse_i = rmse_per_sample(mu, y)
    sigma_reg = (log_sigma ** 2).mean(dim=1)
    return crps_i + (LAMBDA_RMSE * rmse_i) + (LAMBDA_SIGMA_REG * sigma_reg)


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.ema = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.ema[k] = v.detach().to("cpu").clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            v_cpu = v.detach().to("cpu")
            if k not in self.ema:
                self.ema[k] = v_cpu.clone()
                continue
            if torch.is_floating_point(v_cpu):
                self.ema[k].mul_(self.decay).add_(v_cpu, alpha=(1.0 - self.decay))
            else:
                self.ema[k].copy_(v_cpu)

    def state_dict(self):
        return {k: v.clone() for k, v in self.ema.items()}


# ============================================================
# SAM (no GradScaler, autocast ok)
# ============================================================
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.rho = float(rho)
        self.base_optimizer = base_optimizer(params, **kwargs)
        self._e_w = {}

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _grad_norm(self):
        norms = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norms.append(torch.norm(p.grad.detach(), p=2))
        if len(norms) == 0:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        self._e_w.clear()
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                self._e_w[p] = e
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self._e_w.get(p, 0.0))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)


# ============================================================
# Saab Fit (fold into linear)
# ============================================================
@torch.no_grad()
def collect_backbone_features(backbone: nn.Module, dataset: EventDataset,
                              max_samples: int, max_events: int) -> np.ndarray:
    backbone.eval()
    feats = []
    seen = 0
    n_events = min(len(dataset), int(max_events))
    for i in tqdm(range(n_events), desc="collect_saab_feats", leave=False):
        s = dataset[i]
        x = s["images"].to(device, non_blocking=True)
        if amp_enabled:
            with torch.amp.autocast("cuda", enabled=True):
                f = backbone(x).detach().float().cpu().numpy()
        else:
            f = backbone(x).detach().float().cpu().numpy()
        feats.append(f)
        seen += f.shape[0]
        if seen >= max_samples:
            break
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 576), np.float32)
    if X.shape[0] > max_samples:
        X = X[:max_samples]
    return X.astype(np.float32)

def fit_saab_linear(X: np.ndarray, k_dim: int, energy_thr: float, min_eig_ratio: float):
    X = np.asarray(X, np.float32)
    N, D = X.shape
    if N < 16:
        raise RuntimeError(f"Too few samples for Saab fit: N={N}")

    mean = X.mean(axis=0).astype(np.float32)
    Xc = (X - mean).astype(np.float32)

    Xt = torch.from_numpy(Xc)
    q = min(D, max(int(k_dim) + 64, 96))
    q = max(1, min(q, D))

    try:
        _, S, V = torch.pca_lowrank(Xt, q=q, center=False)
        eig = (S ** 2) / max(1.0, float(N - 1))
        V = V[:, :eig.numel()]
    except Exception:
        idx = torch.randperm(N)[: min(N, 12000)]
        Xs = Xt[idx]
        _, S, Vh = torch.linalg.svd(Xs, full_matrices=False)
        V = Vh.transpose(0, 1)
        eig = (S ** 2) / max(1.0, float(Xs.shape[0] - 1))

    eig = eig.detach().cpu().numpy().astype(np.float32)
    V = V.detach().cpu().numpy().astype(np.float32)

    total_var = float(np.maximum(eig.sum(), 1e-12))
    cum = np.cumsum(eig) / total_var
    k0 = int(np.searchsorted(cum, energy_thr) + 1) if np.any(cum >= energy_thr) else len(eig)
    k0 = max(1, min(k0, V.shape[1]))

    keep = []
    for i in range(k0):
        if float(eig[i] / total_var) >= float(min_eig_ratio):
            keep.append(i)
    if len(keep) == 0:
        keep = [0]
    if len(keep) > int(k_dim):
        keep = keep[: int(k_dim)]

    W = V[:, keep].astype(np.float32)
    Z = (Xc @ W).astype(np.float32)
    bias = (-Z.min(axis=0)).astype(np.float32)

    folded_weight = W.T.astype(np.float32)
    folded_bias = (bias - (mean @ W).astype(np.float32)).astype(np.float32)
    return folded_weight, folded_bias


# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def run_eval_loss(model: nn.Module, loader: DataLoader,
                  species_map, domain_map, site_map,
                  y_mean: torch.Tensor, y_std: torch.Tensor) -> float:
    model.eval()
    losses = []
    for batch in loader:
        images_list = [b["images"].to(device, non_blocking=True) for b in batch]
        pri = torch.stack([b["pri"] for b in batch], dim=0).to(device, non_blocking=True)

        sp_idx = torch.tensor([species_map.get(str(b["scientificName"]), 0) for b in batch],
                              dtype=torch.long, device=device)
        dm_idx = torch.tensor([domain_map.get(str(b["domainID"]), 0) for b in batch],
                              dtype=torch.long, device=device)
        st_idx = torch.tensor([site_map.get(str(b["siteID"]), 0) for b in batch],
                              dtype=torch.long, device=device)

        y = torch.stack([b["target"] for b in batch], dim=0).to(device, non_blocking=True)
        y_s = (y - y_mean) / y_std

        if amp_enabled:
            with torch.amp.autocast("cuda", enabled=True):
                mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
                loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()
        else:
            mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
            loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()

        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


# ============================================================
# CALIBRATION: LNT + sigma_scale + var_temp (stronger on 1y/2y)
# ============================================================
@torch.no_grad()
def collect_preds(model: nn.Module, loader: DataLoader,
                  species_map, domain_map, site_map,
                  y_mean: torch.Tensor, y_std: torch.Tensor):
    model.eval()
    preds, sigmas, gts = [], [], []
    for batch in loader:
        images_list = [b["images"].to(device, non_blocking=True) for b in batch]
        pri = torch.stack([b["pri"] for b in batch], dim=0).to(device, non_blocking=True)

        sp_idx = torch.tensor([species_map.get(str(b["scientificName"]), 0) for b in batch],
                              dtype=torch.long, device=device)
        dm_idx = torch.tensor([domain_map.get(str(b["domainID"]), 0) for b in batch],
                              dtype=torch.long, device=device)
        st_idx = torch.tensor([site_map.get(str(b["siteID"]), 0) for b in batch],
                              dtype=torch.long, device=device)

        y = torch.stack([b["target"] for b in batch], dim=0).to(device, non_blocking=True)
        y_s = (y - y_mean) / y_std

        if amp_enabled:
            with torch.amp.autocast("cuda", enabled=True):
                mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
        else:
            mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)

        preds.append(mu_s.detach().float().cpu().numpy())
        sigmas.append(torch.exp(torch.clamp(log_sigma_s, -7.0, 3.0)).detach().float().cpu().numpy())
        gts.append(y_s.detach().float().cpu().numpy())

    X = np.concatenate(preds, axis=0).astype(np.float32)
    S = np.concatenate(sigmas, axis=0).astype(np.float32)
    Y = np.concatenate(gts, axis=0).astype(np.float32)
    return X, S, Y

def fit_lnt_affine(X: np.ndarray, Y: np.ndarray):
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    A = coef[:3, :].astype(np.float32)
    b = coef[3, :].astype(np.float32)
    return A, b

def sigma_scale_from_residuals(X: np.ndarray, S: np.ndarray, Y: np.ndarray, A: np.ndarray, b: np.ndarray):
    Y_hat = X @ A + b
    resid = Y - Y_hat
    resid_rms = np.sqrt(np.mean(resid ** 2, axis=0) + 1e-12)
    sigma_rms = np.sqrt(np.mean(S ** 2, axis=0) + 1e-12)
    sigma_scale = (resid_rms / np.maximum(sigma_rms, 1e-6)).astype(np.float32)

    # boost long horizons a bit for OOD safety
    sigma_scale[1] *= 1.05
    sigma_scale[2] *= 1.10
    return sigma_scale

def fit_variance_temperature(X: np.ndarray, S: np.ndarray, Y: np.ndarray, A: np.ndarray, b: np.ndarray,
                             steps: int = 400, lr: float = 0.05):
    mu = X @ A + b
    sigma = S.copy()

    log_t = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)

    mu_t = torch.from_numpy(mu).float()
    y_t = torch.from_numpy(Y).float()
    sigma_t = torch.from_numpy(sigma).float()

    for _ in range(int(steps)):
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.2, 6.0)
        sig = (sigma_t * t.view(1, 3)).clamp_min(1e-6)

        z = (y_t - mu_t) / sig
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        crps = sig * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))

        # weight long horizons slightly more
        w = torch.tensor([1.0, 1.10, 1.18]).view(1, 3)
        loss = (crps * w).mean()

        loss.backward()
        opt.step()

    t_out = torch.exp(log_t).detach().cpu().numpy().astype(np.float32)
    # extra gentle widen long
    t_out[1] = float(np.clip(t_out[1] * 1.03, 0.2, 6.0))
    t_out[2] = float(np.clip(t_out[2] * 1.06, 0.2, 6.0))
    return t_out


# ============================================================
# model.py (inference) TEXT
# ============================================================
MODEL_PY_TEXT = r'''# -*- coding: utf-8 -*-
"""
model.py (Codabench)
Fast inference:
- MobileNetV3-Small
- Saab folded into saab_linear
- mean+max pooling (event-level)
- multi-head (3 targets)
- Calibration: LNT A,b; sigma_scale; var_temp
- Extra long-horizon sigma floor to reduce overconfidence (helps OOD 1y/2y CRPS)
"""

import os
import io
import math
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

try:
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
    n_threads = max(1, min(n_threads, 8))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)
except Exception:
    pass

WEIGHT_FILENAME = "model_weights.pth"
ENCODERS_FILENAME = "encoders.pkl"
LNT_FILENAME = "lnt_calibration.npz"
EPS = 1e-6

def _torch_load_weights_only_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _open_image(maybe_img: Any, search_roots: List[str], image_side: int) -> Image.Image:
    def black():
        return Image.new("RGB", (image_side, image_side), (0, 0, 0))

    if maybe_img is None:
        return black()

    try:
        if isinstance(maybe_img, Image.Image):
            return maybe_img.convert("RGB")
        if isinstance(maybe_img, (bytes, bytearray)):
            return Image.open(io.BytesIO(maybe_img)).convert("RGB")
        if isinstance(maybe_img, dict):
            if maybe_img.get("bytes") is not None:
                return Image.open(io.BytesIO(maybe_img["bytes"])).convert("RGB")
            p = maybe_img.get("path")
            if isinstance(p, str):
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")
                for r in search_roots:
                    cand = os.path.join(r, p)
                    if os.path.exists(cand):
                        return Image.open(cand).convert("RGB")
            return black()
        if isinstance(maybe_img, str):
            p = maybe_img
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
            for r in search_roots:
                cand = os.path.join(r, p)
                if os.path.exists(cand):
                    return Image.open(cand).convert("RGB")
            return black()
    except Exception:
        return black()

    return black()

class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            m = mobilenet_v3_small(weights=None)
        except Exception:
            m = mobilenet_v3_small(pretrained=False)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x

class FastEventModelMultiHead(nn.Module):
    def __init__(self, k_dim: int, proj_dim: int,
                 species_vocab: int, domain_vocab: int, site_vocab: int,
                 meta_embed_dim: int, meta_dropout: float,
                 pri_dim: int, max_images_per_event: int):
        super().__init__()
        self.max_images_per_event = int(max_images_per_event)

        self.backbone = MobileNetV3SmallBackbone()
        self.saab_linear = nn.Linear(576, int(k_dim), bias=True)

        self.project = nn.Sequential(
            nn.Linear(int(k_dim), int(proj_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.species_emb = nn.Embedding(max(1, int(species_vocab)), int(meta_embed_dim))
        self.domain_emb  = nn.Embedding(max(1, int(domain_vocab)),  int(meta_embed_dim))
        self.site_emb    = nn.Embedding(max(1, int(site_vocab)),    int(meta_embed_dim))
        self.meta_drop = nn.Dropout(float(meta_dropout))

        fused_dim = (2 * int(proj_dim)) + (3 * int(meta_embed_dim)) + 2 + int(pri_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 384),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        def head_block():
            return nn.Sequential(
                nn.Linear(192, 128),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(128, 2),
            )
        self.head_0 = head_block()
        self.head_1 = head_block()
        self.head_2 = head_block()

    def forward(self, images_list: List[torch.Tensor],
                sp_idx: torch.Tensor, dm_idx: torch.Tensor, st_idx: torch.Tensor,
                pri: torch.Tensor):
        dev = next(self.parameters()).device
        lengths = [int(x.shape[0]) for x in images_list]
        x = torch.cat([t.to(dev) for t in images_list], dim=0)

        feats = self.backbone(x).float()
        z = self.saab_linear(feats).float()
        z = self.project(z)

        splits = torch.split(z, lengths, dim=0)
        pooled_list = []
        for f in splits:
            mean = f.mean(dim=0)
            mx = f.max(dim=0).values
            pooled_list.append(torch.cat([mean, mx], dim=0))
        pooled = torch.stack(pooled_list, dim=0)

        sp = self.meta_drop(self.species_emb(sp_idx.to(dev)))
        dm = self.meta_drop(self.domain_emb(dm_idx.to(dev)))
        st = self.meta_drop(self.site_emb(st_idx.to(dev)))

        n_imgs = torch.tensor(lengths, dtype=torch.float32, device=dev).view(-1, 1)
        n_imgs_norm = n_imgs / float(self.max_images_per_event)
        n_imgs_log  = torch.log1p(n_imgs)

        pri = pri.to(dev, dtype=torch.float32)
        fused = torch.cat([pooled, sp, dm, st, n_imgs_norm, n_imgs_log, pri], dim=1)

        h = self.shared(fused)

        o0 = self.head_0(h)
        o1 = self.head_1(h)
        o2 = self.head_2(h)

        mu = torch.stack([o0[:, 0], o1[:, 0], o2[:, 0]], dim=1)
        log_sigma = torch.stack([o0[:, 1], o1[:, 1], o2[:, 1]], dim=1)
        log_sigma = torch.clamp(log_sigma, -7.0, 3.0)
        return mu, log_sigma

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.loaded = False

        self.species_map: Dict[str, int] = {}
        self.domain_map: Dict[str, int] = {}
        self.site_map: Dict[str, int] = {}
        self.priors: Dict[str, Any] = {}

        self.target_mean = np.array([0.0, 0.0, 0.0], np.float32)
        self.target_std  = np.array([1.0, 1.0, 1.0], np.float32)

        self.image_side = 160
        self.max_images_per_event = 4
        self.sigma_floor_long_std_frac = 0.06

        self.lnt_A = None
        self.lnt_b = None
        self.lnt_sigma_scale = None
        self.lnt_var_temp = None

        self._transform = None

    def _candidate_dirs(self) -> List[str]:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        cands = [file_dir]
        if cwd not in cands:
            cands.append(cwd)
        for d in ["/app/ingested_program", "/app/program", "/app"]:
            if d not in cands:
                cands.append(d)
        out = []
        for d in cands:
            out.append(d)
            out.append(os.path.join(d, "artifacts"))
        uniq = []
        for d in out:
            if d not in uniq:
                uniq.append(d)
        return uniq

    def _find_file(self, dirs: List[str], filename: str) -> Optional[str]:
        for d in dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                return p
        return None

    def load(self):
        cands = self._candidate_dirs()
        enc_path = self._find_file(cands, ENCODERS_FILENAME)
        w_path   = self._find_file(cands, WEIGHT_FILENAME)
        lnt_path = self._find_file(cands, LNT_FILENAME)

        if enc_path is None or w_path is None:
            raise RuntimeError(f"Missing artifacts. enc={enc_path}, weights={w_path}. Searched={cands}")

        with open(enc_path, "rb") as f:
            enc = pickle.load(f)

        self.species_map = enc.get("species_map", {}) or {}
        self.domain_map  = enc.get("domain_map", {}) or {}
        self.site_map    = enc.get("site_map", {}) or {}
        self.priors      = enc.get("priors", {}) or {}

        cfg = enc.get("config", {}) if isinstance(enc.get("config", {}), dict) else {}
        self.image_side = int(cfg.get("image_side", self.image_side))
        self.max_images_per_event = int(cfg.get("max_images_per_event", self.max_images_per_event))
        self.sigma_floor_long_std_frac = float(cfg.get("sigma_floor_long_std_frac", self.sigma_floor_long_std_frac))
        pri_dim = int(cfg.get("pri_dim", 12))
        proj_dim = int(cfg.get("proj_dim", 224))
        meta_embed_dim = int(cfg.get("meta_embed_dim", 96))
        meta_dropout = float(cfg.get("meta_dropout", 0.1))
        k_dim = int(cfg.get("saab_k_dim", 192))

        self.target_mean = np.asarray(enc.get("target_mean", self.target_mean), dtype=np.float32)
        self.target_std  = np.asarray(enc.get("target_std",  self.target_std),  dtype=np.float32)

        self._transform = transforms.Compose([
            transforms.Resize((self.image_side, self.image_side)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        sd = _torch_load_weights_only_compat(w_path)

        if "saab_linear.weight" in sd:
            k_dim = int(sd["saab_linear.weight"].shape[0])
        if "project.0.weight" in sd:
            proj_dim = int(sd["project.0.weight"].shape[0])
        if "species_emb.weight" in sd:
            species_vocab = int(sd["species_emb.weight"].shape[0])
            meta_embed_dim = int(sd["species_emb.weight"].shape[1])
        else:
            species_vocab = max(1, len(self.species_map) + 1)
        domain_vocab = int(sd["domain_emb.weight"].shape[0]) if "domain_emb.weight" in sd else max(1, len(self.domain_map) + 1)
        site_vocab   = int(sd["site_emb.weight"].shape[0]) if "site_emb.weight" in sd else max(1, len(self.site_map) + 1)

        self.model = FastEventModelMultiHead(
            k_dim=k_dim,
            proj_dim=proj_dim,
            species_vocab=species_vocab,
            domain_vocab=domain_vocab,
            site_vocab=site_vocab,
            meta_embed_dim=meta_embed_dim,
            meta_dropout=meta_dropout,
            pri_dim=pri_dim,
            max_images_per_event=self.max_images_per_event,
        )

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()

        if lnt_path is not None and os.path.exists(lnt_path):
            z = np.load(lnt_path, allow_pickle=False)
            self.lnt_A = z.get("A", None)
            self.lnt_b = z.get("b", None)
            self.lnt_sigma_scale = z.get("sigma_scale", None)
            self.lnt_var_temp = z.get("var_temp", None)

        self.loaded = True

    def _build_priors_vec(self, scientificName: Any, domainID: Any, siteID: Any) -> torch.Tensor:
        pri = torch.zeros((12,), dtype=torch.float32)
        p = self.priors or {}

        sp_mean = p.get("sp_mean", {}) or {}
        dm_mean = p.get("dm_mean", {}) or {}
        st_mean = p.get("st_mean", {}) or {}
        sp_cnt  = p.get("sp_cnt", {})  or {}
        dm_cnt  = p.get("dm_cnt", {})  or {}
        st_cnt  = p.get("st_cnt", {})  or {}

        sk = str(scientificName)
        dk = str(domainID)
        tk = str(siteID)

        vsp = sp_mean.get(sk, None)
        vdm = dm_mean.get(dk, None)
        vst = st_mean.get(tk, None)

        if vsp is not None: pri[0:3] = torch.from_numpy(np.asarray(vsp, np.float32))
        if vdm is not None: pri[3:6] = torch.from_numpy(np.asarray(vdm, np.float32))
        if vst is not None: pri[6:9] = torch.from_numpy(np.asarray(vst, np.float32))

        pri[9]  = math.log1p(float(sp_cnt.get(sk, 0))) / 6.0
        pri[10] = math.log1p(float(dm_cnt.get(dk, 0))) / 6.0
        pri[11] = math.log1p(float(st_cnt.get(tk, 0))) / 6.0
        return pri

    def _event_to_inputs(self, event: Any):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        roots = [os.getcwd(), file_dir, "/app/input", "/app", "/app/data", "/app/input_data"]

        if isinstance(event, list):
            records = event
        elif isinstance(event, dict):
            records = [event]
        else:
            records = [{}]

        images = []
        keys = ["relative_img", "relative_img_loc", "images", "image_paths", "file_path", "filepath", "path", "image"]
        if isinstance(event, list):
            for d in records:
                val = None
                if isinstance(d, dict):
                    for k in keys:
                        if d.get(k) is not None:
                            val = d.get(k)
                            break
                im = _open_image(val, roots, self.image_side)
                images.append(self._transform(im))
                if len(images) >= self.max_images_per_event:
                    break
        else:
            d0 = records[0] if records else {}
            collected = []
            if isinstance(d0, dict):
                for k in keys:
                    v = d0.get(k)
                    if v is None:
                        continue
                    if isinstance(v, list):
                        collected.extend(v)
                    else:
                        collected.append(v)
            for v in collected:
                im = _open_image(v, roots, self.image_side)
                images.append(self._transform(im))
                if len(images) >= self.max_images_per_event:
                    break

        if len(images) == 0:
            images = [self._transform(Image.new("RGB", (self.image_side, self.image_side), (0, 0, 0)))]

        images_tensor = torch.stack(images, dim=0)

        d0 = records[0] if records else {}
        sci = d0.get("scientificName", None) if isinstance(d0, dict) else None
        dom = d0.get("domainID", None) if isinstance(d0, dict) else None
        site = d0.get("siteID", None) if isinstance(d0, dict) else None

        sp_i = self.species_map.get(str(sci), 0)
        dm_i = self.domain_map.get(str(dom), 0)
        st_i = self.site_map.get(str(site), 0)

        pri = self._build_priors_vec(sci, dom, site).view(1, -1)

        sp_idx = torch.tensor([sp_i], dtype=torch.long, device=self.device)
        dm_idx = torch.tensor([dm_i], dtype=torch.long, device=self.device)
        st_idx = torch.tensor([st_i], dtype=torch.long, device=self.device)

        return [images_tensor.to(self.device)], sp_idx, dm_idx, st_idx, pri.to(self.device)

    def predict(self, event: Any) -> Dict:
        if not self.loaded:
            self.load()
        assert self.model is not None

        images_list, sp_idx, dm_idx, st_idx, pri = self._event_to_inputs(event)

        mean = torch.tensor(self.target_mean, dtype=torch.float32, device=self.device).view(1, 3)
        std  = torch.tensor(self.target_std,  dtype=torch.float32, device=self.device).view(1, 3)

        with torch.inference_mode():
            mu_s, log_sigma_s = self.model(images_list, sp_idx, dm_idx, st_idx, pri)
            mu_s = mu_s.float()
            log_sigma_s = log_sigma_s.float()

            if self.lnt_A is not None and self.lnt_b is not None:
                A = torch.tensor(self.lnt_A, dtype=torch.float32, device=self.device)
                b = torch.tensor(self.lnt_b, dtype=torch.float32, device=self.device).view(1, 3)
                mu_s = mu_s @ A + b

            sigma_s = torch.exp(torch.clamp(log_sigma_s, -7.0, 3.0)).clamp_min(EPS)

            if self.lnt_sigma_scale is not None:
                s = torch.tensor(self.lnt_sigma_scale, dtype=torch.float32, device=self.device).view(1, 3)
                sigma_s = (sigma_s * s).clamp_min(EPS)

            if self.lnt_var_temp is not None:
                t = torch.tensor(self.lnt_var_temp, dtype=torch.float32, device=self.device).view(1, 3)
                sigma_s = (sigma_s * t).clamp_min(EPS)

            # long-horizon sigma floor
            floor = torch.tensor([0.0,
                                  self.sigma_floor_long_std_frac,
                                  self.sigma_floor_long_std_frac],
                                 dtype=torch.float32, device=self.device).view(1, 3)
            sigma_s = torch.maximum(sigma_s, floor)

            mu = (mu_s * std + mean).squeeze(0).detach().cpu().numpy().astype(np.float64).tolist()
            sigma = (sigma_s * std).squeeze(0).detach().cpu().numpy().astype(np.float64).tolist()

        return {
            "SPEI_30d": {"mu": float(mu[0]), "sigma": float(sigma[0])},
            "SPEI_1y":  {"mu": float(mu[1]), "sigma": float(sigma[1])},
            "SPEI_2y":  {"mu": float(mu[2]), "sigma": float(sigma[2])},
        }
'''


# ============================================================
# MAIN TRAIN + PACKAGE
# ============================================================
def main():
    ds = safe_load_hf_dataset(HF_DATASET_NAME)
    if isinstance(ds, dict):
        hf_train = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        hf_val = ds["validation"] if "validation" in ds else None
    else:
        hf_train = ds
        hf_val = None

    print("HF train rows:", len(hf_train))
    if hf_val is not None:
        print("HF validation rows:", len(hf_val))

    train_events_all = build_event_index_hf(hf_train)
    if len(train_events_all) < 50:
        raise RuntimeError("Too few labeled events in train split.")
    print("Train labeled events:", len(train_events_all))

    val_events_all = build_event_index_hf(hf_val) if hf_val is not None else []
    print("Validation labeled events:", len(val_events_all))

    rng = random.Random(SEED)
    by_dom = defaultdict(list)
    for e in train_events_all:
        by_dom[str(e.get("domainID"))].append(e)

    id_val_events, train_events = [], []
    for dom, evs in by_dom.items():
        rng.shuffle(evs)
        n_val = max(1, int(ID_VAL_FRAC_PER_DOMAIN * len(evs)))
        id_val_events.extend(evs[:n_val])
        train_events.extend(evs[n_val:])

    rng.shuffle(train_events)
    rng.shuffle(id_val_events)

    ood_val_events = val_events_all
    if len(ood_val_events) < 10:
        n_ood = max(1, int(0.10 * len(train_events)))
        ood_val_events = train_events[:n_ood]
        train_events = train_events[n_ood:]

    print("Final splits | train:", len(train_events), "ID-val:", len(id_val_events), "OOD-val:", len(ood_val_events))

    species_map, domain_map, site_map = build_encoders(train_events, extra_events=(id_val_events + ood_val_events))
    print("Vocab sizes | species:", len(species_map) + 1, "domain:", len(domain_map) + 1, "site:", len(site_map) + 1)

    y_mean_np, y_std_np = compute_target_stats(train_events)
    print("Target mean:", y_mean_np, "Target std:", y_std_np)
    y_mean = torch.tensor(y_mean_np, dtype=torch.float32, device=device)
    y_std = torch.tensor(y_std_np, dtype=torch.float32, device=device)

    sp_mean, sp_cnt, dm_mean, dm_cnt, st_mean, st_cnt = build_priors(train_events, y_mean_np, y_std_np)
    priors_pack = {"sp_mean": sp_mean, "sp_cnt": sp_cnt,
                   "dm_mean": dm_mean, "dm_cnt": dm_cnt,
                   "st_mean": st_mean, "st_cnt": st_cnt}

    tr_tf, val_tf = make_transforms(IMAGE_SIDE)

    train_ds = EventDataset(train_events, hf_train, tr_tf, MAX_IMAGES_PER_EVENT, shuffle_images=True, priors=priors_pack)
    id_val_ds = EventDataset(id_val_events, hf_train, val_tf, MAX_IMAGES_PER_EVENT, shuffle_images=False, priors=priors_pack)

    ood_hf = hf_val if (hf_val is not None and len(val_events_all) >= 10) else hf_train
    ood_val_ds = EventDataset(ood_val_events, ood_hf, val_tf, MAX_IMAGES_PER_EVENT, shuffle_images=False, priors=priors_pack)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_events,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    id_val_loader = DataLoader(id_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_events,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    ood_val_loader = DataLoader(ood_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_events,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = FastEventModelMultiHead(
        k_dim=SAAB_K_DIM,
        proj_dim=224,
        species_vocab=(len(species_map) + 1),
        domain_vocab=(len(domain_map) + 1),
        site_vocab=(len(site_map) + 1),
        meta_embed_dim=META_EMBED_DIM,
        meta_dropout=META_DROPOUT,
        pri_dim=PRI_DIM,
        max_images_per_event=MAX_IMAGES_PER_EVENT,
        backbone_pretrained=True,
        freeze_backbone=True,
    ).to(device)

    # Fit Saab folded
    print("\nFitting folded Saab ...")
    saab_fit_ds = EventDataset(train_events, hf_train, val_tf, MAX_IMAGES_PER_EVENT, shuffle_images=False, priors=priors_pack)
    X = collect_backbone_features(model.backbone.to(device), saab_fit_ds,
                                  max_samples=SAAB_FIT_MAX_SAMPLES, max_events=SAAB_FIT_MAX_EVENTS)
    print("Backbone feature matrix:", X.shape)

    folded_w, folded_b = fit_saab_linear(X, SAAB_K_DIM, SAAB_ENERGY_THRESHOLD, SAAB_MIN_EIG_RATIO)
    with torch.no_grad():
        model.saab_linear.weight.copy_(torch.from_numpy(folded_w))
        model.saab_linear.bias.copy_(torch.from_numpy(folded_b))
    print("Saab folded:", model.saab_linear.weight.shape)

    def make_param_groups(unfreeze_backbone: bool):
        groups = []
        if unfreeze_backbone:
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = [p for n, p in model.named_parameters() if (p.requires_grad and not n.startswith("backbone."))]
            if backbone_params:
                groups.append({"params": backbone_params, "lr": LR * BACKBONE_LR_MULT})
            if head_params:
                groups.append({"params": head_params, "lr": LR})
        else:
            head_params = [p for p in model.parameters() if p.requires_grad]
            groups.append({"params": head_params, "lr": LR})
        return groups

    if USE_SAM:
        optimizer = SAM(make_param_groups(False), base_optimizer=optim.AdamW, rho=SAM_RHO, weight_decay=WEIGHT_DECAY)
        sched_optim = optimizer.base_optimizer
    else:
        optimizer = optim.AdamW(make_param_groups(False), lr=LR, weight_decay=WEIGHT_DECAY)
        sched_optim = optimizer

    scheduler = optim.lr_scheduler.CosineAnnealingLR(sched_optim, T_max=EPOCHS, eta_min=LR * 0.05)
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_score = float("inf")
    best_state = None

    print("\nBegin training...")
    for epoch in range(EPOCHS):

        if UNFREEZE_BACKBONE_AT_EPOCH is not None and epoch == int(UNFREEZE_BACKBONE_AT_EPOCH):
            print(f"\n[INFO] Unfreezing backbone at epoch {epoch} ...")
            for p in model.backbone.parameters():
                p.requires_grad_(True)

            if USE_SAM:
                optimizer = SAM(make_param_groups(True), base_optimizer=optim.AdamW, rho=SAM_RHO, weight_decay=WEIGHT_DECAY)
                sched_optim = optimizer.base_optimizer
            else:
                optimizer = optim.AdamW(make_param_groups(True), lr=LR, weight_decay=WEIGHT_DECAY)
                sched_optim = optimizer

            scheduler = optim.lr_scheduler.CosineAnnealingLR(sched_optim, T_max=(EPOCHS - epoch), eta_min=LR * 0.05)

        model.train()
        running = 0.0
        nb = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            images_list = [b["images"].to(device, non_blocking=True) for b in batch]
            pri = torch.stack([b["pri"] for b in batch], dim=0).to(device, non_blocking=True)

            sp_idx = torch.tensor([species_map.get(str(b["scientificName"]), 0) for b in batch],
                                  dtype=torch.long, device=device)
            dm_idx = torch.tensor([domain_map.get(str(b["domainID"]), 0) for b in batch],
                                  dtype=torch.long, device=device)
            st_idx = torch.tensor([site_map.get(str(b["siteID"]), 0) for b in batch],
                                  dtype=torch.long, device=device)

            if META_DROP_PROB_TRAIN > 0:
                m = (torch.rand_like(sp_idx.float()) < META_DROP_PROB_TRAIN)
                sp_idx = torch.where(m, torch.zeros_like(sp_idx), sp_idx)
                m = (torch.rand_like(dm_idx.float()) < META_DROP_PROB_TRAIN)
                dm_idx = torch.where(m, torch.zeros_like(dm_idx), dm_idx)
                m = (torch.rand_like(st_idx.float()) < META_DROP_PROB_TRAIN)
                st_idx = torch.where(m, torch.zeros_like(st_idx), st_idx)

            y = torch.stack([b["target"] for b in batch], dim=0).to(device, non_blocking=True)
            y_s = (y - y_mean) / y_std

            if USE_SAM:
                # SAM step 1
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    with torch.amp.autocast("cuda", enabled=True):
                        mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
                        loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()
                else:
                    mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
                    loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.first_step(zero_grad=True)

                # SAM step 2
                if amp_enabled:
                    with torch.amp.autocast("cuda", enabled=True):
                        mu_s2, log_sigma_s2 = model(images_list, sp_idx, dm_idx, st_idx, pri)
                        loss2 = total_loss_per_sample(mu_s2, log_sigma_s2, y_s).mean()
                else:
                    mu_s2, log_sigma_s2 = model(images_list, sp_idx, dm_idx, st_idx, pri)
                    loss2 = total_loss_per_sample(mu_s2, log_sigma_s2, y_s).mean()

                if not torch.isfinite(loss2):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss2.backward()
                optimizer.second_step(zero_grad=True)

                loss_val = float(loss2.item())

            else:
                # Standard AdamW + GradScaler
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    with torch.amp.autocast("cuda", enabled=True):
                        mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
                        loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()
                else:
                    mu_s, log_sigma_s = model(images_list, sp_idx, dm_idx, st_idx, pri)
                    loss = total_loss_per_sample(mu_s, log_sigma_s, y_s).mean()

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if scaler_enabled:
                    scaler.scale(loss).backward()
                    if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

                loss_val = float(loss.item())

            if ema is not None:
                ema.update(model)

            running += loss_val
            nb += 1
            pbar.set_postfix({"loss": running / max(1, nb)})

        scheduler.step()

        # eval EMA
        if ema is not None:
            tmp = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(), strict=True)
            model.to(device).eval()
            id_loss = run_eval_loss(model, id_val_loader, species_map, domain_map, site_map, y_mean, y_std)
            ood_loss = run_eval_loss(model, ood_val_loader, species_map, domain_map, site_map, y_mean, y_std)
            model.load_state_dict(tmp, strict=True)
            model.to(device)
        else:
            id_loss = run_eval_loss(model, id_val_loader, species_map, domain_map, site_map, y_mean, y_std)
            ood_loss = run_eval_loss(model, ood_val_loader, species_map, domain_map, site_map, y_mean, y_std)

        combined = 0.65 * ood_loss + 0.35 * id_loss
        print(f"Epoch {epoch+1}: ID={id_loss:.4f} OOD={ood_loss:.4f} combined={combined:.4f}")

        if combined < best_score:
            best_score = combined
            best_state = ema.state_dict() if (ema is not None) else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  -> New BEST")

    if best_state is None:
        raise RuntimeError("Training did not produce best_state.")

    torch.save(best_state, str(WEIGHTS_PATH))
    print("Saved:", WEIGHTS_PATH)

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({
            "species_map": species_map,
            "domain_map": domain_map,
            "site_map": site_map,
            "priors": priors_pack,
            "target_mean": y_mean_np.astype(np.float32),
            "target_std": y_std_np.astype(np.float32),
            "config": {
                "image_side": int(IMAGE_SIDE),
                "max_images_per_event": int(MAX_IMAGES_PER_EVENT),
                "pri_dim": int(PRI_DIM),
                "proj_dim": 224,
                "meta_embed_dim": int(META_EMBED_DIM),
                "meta_dropout": float(META_DROPOUT),
                "saab_k_dim": int(SAAB_K_DIM),
                "sigma_floor_long_std_frac": float(SIGMA_FLOOR_LONG_STD_FRAC),
            }
        }, f)
    print("Saved:", ENCODERS_PATH)

    # Calibration
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu"), strict=True)
    model.to(device).eval()

    Xp, Sp, Yp = collect_preds(model, id_val_loader, species_map, domain_map, site_map, y_mean, y_std)
    A, b = fit_lnt_affine(Xp, Yp)
    sigma_scale = sigma_scale_from_residuals(Xp, Sp, Yp, A, b)

    var_temp = np.ones((3,), np.float32)
    if USE_VARIANCE_TEMP:
        Sp2 = Sp * sigma_scale.reshape(1, 3)
        var_temp = fit_variance_temperature(Xp, Sp2, Yp, A, b, steps=VT_STEPS, lr=VT_LR)

    np.savez_compressed(str(LNT_PATH), A=A, b=b, sigma_scale=sigma_scale.astype(np.float32), var_temp=var_temp.astype(np.float32))
    print("Saved:", LNT_PATH, "| sigma_scale:", sigma_scale, "| var_temp:", var_temp)

    # Write model.py + empty requirements
    (OUT_DIR / "model.py").write_text(MODEL_PY_TEXT, encoding="utf-8")
    (OUT_DIR / "requirements.txt").write_text("", encoding="utf-8")
    (OUT_DIR / "requirements.txt.txt").write_text("", encoding="utf-8")

    # ZIP root-level
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ["model.py", WEIGHT_FILENAME, ENCODERS_FILENAME, LNT_FILENAME, "requirements.txt", "requirements.txt.txt"]:
            p = OUT_DIR / fn
            if not p.exists():
                raise RuntimeError(f"Missing file for zip: {p}")
            z.write(str(p), arcname=fn)

    print("\nDONE. Upload:", ZIP_PATH.resolve())
    with zipfile.ZipFile(str(ZIP_PATH), "r") as z:
        print("Zip content:")
        for n in z.namelist():
            print(" -", n)


if __name__ == "__main__":
    main()
