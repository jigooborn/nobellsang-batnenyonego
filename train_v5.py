# -*- coding: utf-8 -*-
"""
train_v5.py — v5 4-태스크 학습: 6분류(OB~M) + Teff + logg + 광도계급

v4 레시피 계승 (전부 검증된 설정):
  effective-number 가중 샘플링 / EMA 0.997 / warmup3+cosine / AMP /
  실시간 증강 3종 / best 선택 = macro 재현율 / seed 42

v4 → v5 변경:
  1) 클래스 7→6 (OB 통합, 교과서 경계 — preprocess_core 에서 import)
  2) 피처 16→39 (EW 27 + FWHM 5 + 코어/날개 5 + 색지수 2)
     feat_norm_v5.npy(train 통계)로 표준화해 입력 (GUI 도 동일 적용)
  3) 광도계급(거성/주계열/백색왜성) 전용 분류 헤드 추가 — v4 는 예측
     logg 구간 컷이었는데 백색왜성(표본 288)이 회귀 평균쏠림으로 4%만
     잡혔음 → 가중 CE 분류 헤드로 교체. logg 회귀는 물리량 출력으로 유지
  4) CNN 에 색지수(cont_slope/curve) 주입 — 연속선 정규화로 사라진 '개형'
     정보를 CNN 헤드가 직접 보게 함 (풀링 256 + 색지수 2 = 258)

실행: python train_v5.py   (v5 폴더, data/ 생성 후)
출력: models/resnet_v5.pth, models/mlp_v5.pth, results/run_NNN/
"""

import os
import copy
import math
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib

from preprocess_core import (CLASS_ORDER, N_CLASSES, N_FEATURES,
                             LUM_ORDER, N_LUM)

SEED          = 42
BATCH_SIZE    = 256
EPOCHS        = 60
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.05
EMA_DECAY     = 0.997
EMA_BURN_IN   = 5
ENS_BETA      = 0.99
LAMBDA_TEFF   = 0.5
LAMBDA_LOGG   = 0.3
LAMBDA_LUM    = 0.3      # 광도계급 헤드 (가중 CE — wd 표본 부족 보정)
SELECT_METRIC = "macro"

LUM_TO_IDX = {c: i for i, c in enumerate(LUM_ORDER)}
N_COLOR = 2              # CNN 에 주입하는 색지수 개수 (피처 마지막 2개)

AUG_NOISE_MAX = 0.04
AUG_TILT_MAX  = 0.05
AUG_SHIFT_PIX = 2
AUG_FEAT_SD   = 0.01   # 표준화 공간에서의 지터

CLS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = "data"


def _init_env():
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rc('font', family='Malgun Gothic')
    except Exception:
        pass
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True


class SpecDatasetV5(Dataset):
    """mmap 행 단위 접근 (저메모리). 피처는 train 통계로 표준화.
    반환: flux, feat_std, 분광형, teff_n, logg_n, logg_ok, lum, lum_ok"""

    def __init__(self, flux_mm, feat_mm, idx, labels, teff_n, logg_n,
                 logg_ok, lum_idx, lum_ok, feat_mu, feat_sd, augment=False):
        self.flux = flux_mm
        self.feat = feat_mm
        self.idx = idx.astype(np.int64)
        self.labels = torch.from_numpy(labels[idx]).long()
        self.teff = torch.from_numpy(teff_n[idx]).float()
        self.logg = torch.from_numpy(logg_n[idx]).float()
        self.lok = torch.from_numpy(logg_ok[idx]).float()
        self.lum = torch.from_numpy(lum_idx[idx]).long()
        self.lumok = torch.from_numpy(lum_ok[idx]).float()
        self.fmu = feat_mu.astype(np.float32)
        self.fsd = feat_sd.astype(np.float32)
        self.augment = augment
        self.n_pix = flux_mm.shape[1]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        r = self.idx[k]
        f = np.array(self.flux[r], dtype=np.float32)
        x = (np.array(self.feat[r], dtype=np.float32) - self.fmu) / self.fsd
        if self.augment:
            s = np.random.randint(-AUG_SHIFT_PIX, AUG_SHIFT_PIX + 1)
            if s != 0:
                f = np.roll(f, s)
                if s > 0: f[:s] = 1.0
                else:     f[s:] = 1.0
            a = np.random.uniform(-AUG_TILT_MAX, AUG_TILT_MAX)
            f = f * (1.0 + a * np.linspace(-1.0, 1.0, self.n_pix,
                                           dtype=np.float32))
            sig = np.random.uniform(0.0, AUG_NOISE_MAX)
            if sig > 0:
                f = f + np.random.normal(0.0, sig, self.n_pix
                                         ).astype(np.float32)
            f = np.clip(f, 0.0, 5.0)
            x = x + np.random.normal(0.0, AUG_FEAT_SD, len(x)
                                     ).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(f)),
                torch.from_numpy(np.ascontiguousarray(x)),
                self.labels[k], self.teff[k], self.logg[k], self.lok[k],
                self.lum[k], self.lumok[k])


# ── 모델 (v4 구조 계승, 클래스 수만 6) ──────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.GELU(),
            nn.Linear(ch // r, ch), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x.mean(dim=2))
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.se = SEBlock(c_out)
        self.skip = (nn.Identity() if c_in == c_out
                     else nn.Conv1d(c_in, c_out, 1))
        self.act = nn.GELU()
    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return self.act(h + self.skip(x))


class AttnPool(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        w = torch.softmax(self.score(x), dim=2)
        return (x * w).sum(dim=2)


class SpectralResNetV5(nn.Module):
    """4-태스크: 분광형 + Teff + logg + 광도계급.
    v5: 어텐션 풀링(256) + 색지수 2개(피처 뒤 2개, 표준화됨)를 이어붙여
    헤드 입력 258 — 연속선 정규화로 사라진 '개형' 정보를 직접 공급."""
    def __init__(self, n_classes: int = N_CLASSES, n_lum: int = N_LUM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(ResBlock(32, 64, k=11), nn.MaxPool1d(4))
        self.stage2 = nn.Sequential(ResBlock(64, 128, k=7), nn.MaxPool1d(4))
        self.stage3 = nn.Sequential(ResBlock(128, 256, k=5), nn.MaxPool1d(2))
        self.pool = AttnPool(256)
        d = 256 + N_COLOR
        self.cls_head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))
        self.teff_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.logg_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.lum_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_lum))
    def forward(self, flux, feat):
        z = self.stem(flux.unsqueeze(1))
        z = self.stage3(self.stage2(self.stage1(z)))
        z = self.pool(z)
        z = torch.cat([z, feat[:, -N_COLOR:]], dim=1)   # 색지수 주입
        return (self.cls_head(z), self.teff_head(z).squeeze(-1),
                self.logg_head(z).squeeze(-1), self.lum_head(z))


class FeatureMLPV5(nn.Module):
    def __init__(self, n_features: int = N_FEATURES,
                 n_classes: int = N_CLASSES, n_lum: int = N_LUM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes))
        self.teff_net = nn.Sequential(
            nn.Linear(n_features, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.logg_net = nn.Sequential(
            nn.Linear(n_features, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.lum_net = nn.Sequential(
            nn.Linear(n_features, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, n_lum))
    def forward(self, flux, feat):
        return (self.net(feat), self.teff_net(feat).squeeze(-1),
                self.logg_net(feat).squeeze(-1), self.lum_net(feat))


class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.active = False
    def activate(self, model):
        es = self.ema.state_dict()
        for k, v in model.state_dict().items():
            es[k].copy_(v)
        self.active = True
    @torch.no_grad()
    def update(self, model):
        if not self.active:
            return
        es = self.ema.state_dict()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                es[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                es[k].copy_(v)


def effective_num_weights(counts, beta=ENS_BETA):
    counts = np.maximum(counts.astype(np.float64), 1.0)
    w = 1.0 / ((1.0 - np.power(beta, counts)) / (1.0 - beta))
    return (w / w.sum() * len(counts)).astype(np.float32)


def lr_factor(e, total, warmup):
    if e < warmup:
        return (e + 1) / max(warmup, 1)
    p = (e - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_model(model, loader_tr, loader_val, name, save_path,
                teff_sd=1.0, logg_sd=1.0, lum_weights=None):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: lr_factor(e, EPOCHS, WARMUP_EPOCHS))
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    # 광도계급: wd 표본 부족 → 클래스 가중 CE (reduction none — lum_ok 마스크)
    lw = (torch.ones(N_LUM) if lum_weights is None
          else torch.from_numpy(lum_weights).float()).to(DEVICE)
    crit_lum = nn.CrossEntropyLoss(weight=lw, reduction='none')
    huber = nn.SmoothL1Loss(reduction='none')
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    best_score, best_kind = -1.0, '?'
    hist = {'tr_loss': [], 'val_acc_raw': [], 'val_acc_ema': [],
            'val_teff_err': [], 'val_logg_err': [], 'val_macro': [],
            'val_lum_macro': []}

    def evaluate(net):
        net.eval()
        c = t = 0
        dteff = dlogg = nlogg = 0.0
        per_cls = np.zeros((N_CLASSES, 2))
        per_lum = np.zeros((N_LUM, 2))
        with torch.no_grad():
            for fb, xb, lb, tb, gb, ok, ub, uok in loader_val:
                fb, xb = fb.to(DEVICE), xb.to(DEVICE)
                lb, tb = lb.to(DEVICE), tb.to(DEVICE)
                gb, ok = gb.to(DEVICE), ok.to(DEVICE)
                ub, uok = ub.to(DEVICE), uok.to(DEVICE)
                logits, tp, gp, lu = net(fb, xb)
                pred = logits.argmax(1)
                c += (pred == lb).sum().item()
                t += lb.size(0)
                dteff += (tp - tb).abs().sum().item() * teff_sd
                dlogg += ((gp - gb).abs() * ok).sum().item() * logg_sd
                nlogg += ok.sum().item()
                for k in range(N_CLASSES):
                    m = lb == k
                    per_cls[k, 0] += (pred[m] == k).sum().item()
                    per_cls[k, 1] += m.sum().item()
                lpred = lu.argmax(1)
                for k in range(N_LUM):
                    m = (ub == k) & (uok > 0)
                    per_lum[k, 0] += (lpred[m] == k).sum().item()
                    per_lum[k, 1] += m.sum().item()
        rec = per_cls[:, 0] / np.maximum(per_cls[:, 1], 1)
        macro = float(rec[per_cls[:, 1] > 0].mean())
        lrec = per_lum[:, 0] / np.maximum(per_lum[:, 1], 1)
        lum_macro = float(lrec[per_lum[:, 1] > 0].mean())
        return (c / t, dteff / t, dlogg / max(nlogg, 1), macro,
                lum_macro, per_cls, per_lum)

    per_cls = per_lum = None
    for epoch in range(1, EPOCHS + 1):
        if epoch == EMA_BURN_IN + 1 and not ema.active:
            ema.activate(model)

        model.train()
        tr_loss = 0.0
        for fb, xb, lb, tb, gb, ok, ub, uok in loader_tr:
            fb, xb = fb.to(DEVICE), xb.to(DEVICE)
            lb, tb = lb.to(DEVICE), tb.to(DEVICE)
            gb, ok = gb.to(DEVICE), ok.to(DEVICE)
            ub, uok = ub.to(DEVICE), uok.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                logits, tp, gp, lu = model(fb, xb)
                loss = crit(logits, lb)
                loss = loss + LAMBDA_TEFF * huber(tp, tb).mean()
                lg = (huber(gp, gb) * ok).sum() / ok.sum().clamp(min=1)
                loss = loss + LAMBDA_LOGG * lg
                ll = (crit_lum(lu, ub) * uok).sum() / uok.sum().clamp(min=1)
                loss = loss + LAMBDA_LUM * ll
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tr_loss += loss.item()
        sched.step()

        acc_raw, terr_r, gerr_r, mac_raw, lmac_r, _, _ = evaluate(model)
        if ema.active:
            (acc_ema, terr_e, gerr_e, mac_ema, lmac_e,
             per_cls, per_lum) = evaluate(ema.ema)
        else:
            acc_ema, terr_e, gerr_e, mac_ema, lmac_e = 0, 0, 0, 0, 0

        hist['tr_loss'].append(tr_loss / len(loader_tr))
        hist['val_acc_raw'].append(acc_raw)
        hist['val_acc_ema'].append(acc_ema)
        hist['val_teff_err'].append(terr_e if ema.active else terr_r)
        hist['val_logg_err'].append(gerr_e if ema.active else gerr_r)
        hist['val_macro'].append(max(mac_raw, mac_ema))
        hist['val_lum_macro'].append(max(lmac_r, lmac_e))

        if SELECT_METRIC == "macro":
            s_raw, s_ema = mac_raw, mac_ema
        else:
            s_raw, s_ema = acc_raw, acc_ema
        cand = max(s_raw, s_ema)
        kind = 'EMA' if s_ema > s_raw else 'raw'
        if cand > best_score:
            best_score, best_kind = cand, kind
            state = (ema.ema.state_dict() if kind == 'EMA'
                     else model.state_dict())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(state, save_path)

        if epoch % 5 == 0 or epoch <= 3:
            print(f"  [{name}] ep {epoch:3d}/{EPOCHS}"
                  f" | loss {hist['tr_loss'][-1]:.4f}"
                  f" | acc {max(acc_raw, acc_ema):.4f}"
                  f" | macro {max(mac_raw, mac_ema):.4f}"
                  f" | lum {hist['val_lum_macro'][-1]:.4f}"
                  f" | Teff {hist['val_teff_err'][-1]:.4f}dex"
                  f" | logg {hist['val_logg_err'][-1]:.3f}"
                  f" | best {best_score:.4f}({best_kind})", flush=True)

    print(f"  [{name}] 최고 {SELECT_METRIC}={best_score:.4f} ({best_kind})")
    if per_cls is not None:
        rec = ["{}:{:.0f}%".format(CLASS_ORDER[k],
               100 * per_cls[k, 0] / max(per_cls[k, 1], 1))
               for k in range(N_CLASSES)]
        print(f"  [{name}] (마지막 EMA) 재현율: " + "  ".join(rec))
        lrec = ["{}:{:.0f}%".format(LUM_ORDER[k],
                100 * per_lum[k, 0] / max(per_lum[k, 1], 1))
                for k in range(N_LUM)]
        print(f"  [{name}] 광도계급 재현율: " + "  ".join(lrec))
    return hist


def make_run_dir():
    os.makedirs("results", exist_ok=True)
    nums = [int(d.split('_')[1]) for d in os.listdir("results")
            if d.startswith("run_") and d.split('_')[1].isdigit()]
    run = max(nums, default=0) + 1
    path = os.path.join("results", f"run_{run:03d}")
    os.makedirs(path, exist_ok=True)
    return path, run


def main():
    _init_env()
    import matplotlib.pyplot as plt
    run_dir, run_num = make_run_dir()
    print("=" * 66)
    print(f"v5 학습 — 6클래스(OB~M) 4-태스크(분류+Teff+logg+광도계급), "
          f"피처 {N_FEATURES}개(표준화)")
    print(f"[device: {DEVICE}, AMP: {DEVICE.type=='cuda'}, "
          f"best={SELECT_METRIC}]  run_{run_num:03d}")
    print("=" * 66)

    flux = np.load(f"{DATA}/v5_train_flux.npy", mmap_mode='r')
    feat = np.load(f"{DATA}/v5_train_features.npy", mmap_mode='r')
    df = pd.read_csv(f"{DATA}/v5_train_labels.csv")
    labels = np.array([CLS_TO_IDX[l] for l in df['LABEL']], dtype=np.int64)

    lt_mu, lt_sd = np.load(f"{DATA}/teff_norm_v5.npy")
    teff_n = ((np.log10(df['TEFF'].values) - lt_mu) / lt_sd
              ).astype(np.float32)
    lg_mu, lg_sd = np.load(f"{DATA}/logg_norm_v5.npy")
    logg = df['LOGG'].values.astype(np.float64)
    ok = np.isfinite(logg)
    logg_n = np.where(ok, (logg - lg_mu) / lg_sd, 0.0).astype(np.float32)
    logg_ok = ok.astype(np.float32)
    fstats = np.load(f"{DATA}/feat_norm_v5.npy")
    f_mu, f_sd = fstats[0], fstats[1]

    # 광도계급 라벨 (giant/ms/wd — unknown 은 lum_ok=0 으로 손실 제외)
    lum_idx = np.array([LUM_TO_IDX.get(l, 0) for l in df['LUM']],
                       dtype=np.int64)
    lum_ok = np.array([1.0 if l in LUM_TO_IDX else 0.0
                       for l in df['LUM']], dtype=np.float32)
    lum_counts = np.array([(df['LUM'] == c).sum() for c in LUM_ORDER])
    # β=0.99 는 수백 개 이상에서 포화(전부 1.0) → 광도계급은 β=0.9999 로
    # 불균형(주계열 7.7만 vs 백색왜성 1.2천)이 실제로 반영되게 함
    lum_w = effective_num_weights(lum_counts, beta=0.9999)
    print(f"train: {flux.shape} (mmap), logg 라벨 {int(ok.sum()):,}"
          f"/{len(df):,}")
    print("분포:", {c: int((labels == i).sum())
                    for i, c in enumerate(CLASS_ORDER)})
    print("광도계급:", dict(zip(LUM_ORDER, lum_counts.tolist())),
          " 가중치:", np.round(lum_w, 2).tolist())

    from sklearn.model_selection import train_test_split
    idx_tr, idx_val = train_test_split(np.arange(len(df)), test_size=0.15,
                                       random_state=SEED, stratify=labels)

    cw = effective_num_weights(
        np.bincount(labels[idx_tr], minlength=N_CLASSES))
    sampler = WeightedRandomSampler(
        torch.from_numpy(cw[labels[idx_tr]]).float(),
        num_samples=len(idx_tr), replacement=True)

    ds_tr = SpecDatasetV5(flux, feat, idx_tr, labels, teff_n, logg_n,
                          logg_ok, lum_idx, lum_ok, f_mu, f_sd,
                          augment=True)
    ds_va = SpecDatasetV5(flux, feat, idx_val, labels, teff_n, logg_n,
                          logg_ok, lum_idx, lum_ok, f_mu, f_sd,
                          augment=False)
    l_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=0, pin_memory=True)
    l_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)

    print(f"\n[1/2] SpectralResNetV5 ({EPOCHS} epochs)...")
    cnn = SpectralResNetV5().to(DEVICE)
    print(f"  파라미터: {sum(p.numel() for p in cnn.parameters()):,}")
    h_cnn = train_model(cnn, l_tr, l_va, "resnet5",
                        os.path.join("models", "resnet_v5.pth"),
                        teff_sd=float(lt_sd), logg_sd=float(lg_sd),
                        lum_weights=lum_w)

    print(f"\n[2/2] FeatureMLPV5 ({EPOCHS} epochs)...")
    mlp = FeatureMLPV5().to(DEVICE)
    print(f"  파라미터: {sum(p.numel() for p in mlp.parameters()):,}")
    h_mlp = train_model(mlp, l_tr, l_va, "mlp5",
                        os.path.join("models", "mlp_v5.pth"),
                        teff_sd=float(lt_sd), logg_sd=float(lg_sd),
                        lum_weights=lum_w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, h, title in zip(axes, [h_cnn, h_mlp],
                            ['SpectralResNetV5', 'FeatureMLPV5']):
        ep = range(1, len(h['tr_loss']) + 1)
        ax2 = ax.twinx()
        ax.plot(ep, h['tr_loss'], color='#1f77b4', lw=1.5, label='loss')
        ax2.plot(ep, h['val_acc_raw'], color='#ff7f0e', lw=1, alpha=0.5,
                 label='acc(raw)')
        ax2.plot(ep, h['val_macro'], color='#d62728', lw=1.8,
                 label='macro recall')
        ax2.plot(ep, h['val_lum_macro'], color='#8c564b', lw=1.2,
                 label='lum macro')
        ax2.plot(ep, h['val_teff_err'], color='#2ca02c', lw=1.2, ls='--',
                 label='Teff err (dex)')
        ax2.plot(ep, np.array(h['val_logg_err']) / 10, color='#9467bd',
                 lw=1.2, ls=':', label='logg err (/10)')
        ax.set_xlabel('epoch'); ax.set_title(title)
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, loc='center right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curves_v5.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    with open(os.path.join(run_dir, "run_info.txt"), 'w',
              encoding='utf-8') as f:
        f.write(f"v5 run {run_num} @ {datetime.datetime.now()}\n")
        f.write(f"6클래스(OB~M 교과서 경계), 4-task(+광도 거성/주계열/WD), "
                f"피처 {N_FEATURES}개, best={SELECT_METRIC}\n")
        f.write(f"lambda_teff={LAMBDA_TEFF} lambda_logg={LAMBDA_LOGG} "
                f"lambda_lum={LAMBDA_LUM}\n")
        f.write(f"resnet macro_best={max(h_cnn['val_macro']):.4f}\n")
        f.write(f"mlp    macro_best={max(h_mlp['val_macro']):.4f}\n")

    print(f"\n완료 → models/resnet_v5.pth, models/mlp_v5.pth, {run_dir}/")
    print("다음: python eval_v5.py")


if __name__ == "__main__":
    main()
