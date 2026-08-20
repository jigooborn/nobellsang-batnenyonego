# -*- coding: utf-8 -*-
"""
kfold_v5.py — v5 파이프라인 k-fold 교차 검증 (별 그룹 단위)

kfold_v4 와 동일한 절차. 변경: 6클래스 / 39피처(표준화) / v5 파일명.
- 폴드 분할: GROUP 단위 → 같은 별이 학습/검증 양쪽에 못 감
- 증강 행(#augN)은 원본과 같은 그룹 → 같은 폴드, 검증은 원본만
실행: python kfold_v5.py            (K=5, 60 epochs — 밤새 규모)
      python kfold_v5.py --epochs 30   (빠른 예비)
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_v5 as T
from train_v5 import (SpectralResNetV5, FeatureMLPV5, SpecDatasetV5,
                      train_model, effective_num_weights,
                      CLS_TO_IDX, LUM_TO_IDX, BATCH_SIZE, DEVICE, _init_env)
from preprocess_core import CLASS_ORDER, N_CLASSES, LUM_ORDER

OUT_DIR = os.path.join("results", "kfold_v5")
TMP_DIR = os.path.join("models", "kfold_v5_tmp")


def load_pool():
    parts_f, parts_x, parts_df = [], [], []
    for tag in ["train", "test"]:
        parts_f.append(np.load(f"data/v5_{tag}_flux.npy"))
        parts_x.append(np.load(f"data/v5_{tag}_features.npy"))
        d = pd.read_csv(f"data/v5_{tag}_labels.csv")
        if "AUG" not in d.columns:
            d["AUG"] = 0
        parts_df.append(d)
    flux = np.concatenate(parts_f).astype(np.float32)
    feat = np.concatenate(parts_x).astype(np.float32)
    df = pd.concat(parts_df, ignore_index=True)
    labels = np.array([CLS_TO_IDX[l] for l in df["LABEL"]], dtype=np.int64)
    lt_mu, lt_sd = np.load("data/teff_norm_v5.npy")
    teff_n = ((np.log10(df["TEFF"].values) - lt_mu) / lt_sd
              ).astype(np.float32)
    logg = df["LOGG"].values.astype(np.float64)
    ok = np.isfinite(logg)
    lg_mu, lg_sd = np.load("data/logg_norm_v5.npy")
    logg_n = np.where(ok, (logg - lg_mu) / lg_sd, 0.0).astype(np.float32)
    logg_ok = ok.astype(np.float32)
    lum_idx = np.array([LUM_TO_IDX.get(l, 0) for l in df['LUM']],
                       dtype=np.int64)
    lum_ok = np.array([1.0 if l in LUM_TO_IDX else 0.0
                       for l in df['LUM']], dtype=np.float32)
    fstats = np.load("data/feat_norm_v5.npy")
    return (flux, feat, labels, teff_n, logg_n, logg_ok, lum_idx, lum_ok,
            df, float(lt_sd), float(lg_sd), fstats[0], fstats[1])


@torch.no_grad()
def probs_of(model, flux, feat_std):
    model.eval()
    out = np.zeros((len(flux), N_CLASSES))
    B = 1024
    for i in range(0, len(flux), B):
        fb = torch.from_numpy(flux[i:i+B]).float().to(DEVICE)
        xb = torch.from_numpy(feat_std[i:i+B]).float().to(DEVICE)
        lg, _, _, _ = model(fb, xb)
        out[i:i+B] = torch.softmax(lg, 1).cpu().numpy()
    return out


def metrics(pred, y):
    acc = float((pred == y).mean())
    rec = np.array([(pred[y == k] == k).mean() if (y == k).any() else np.nan
                    for k in range(N_CLASSES)])
    return acc, float(np.nanmean(rec)), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only-folds", default=None,
                    help="지정 폴드만 실행 (예: 4,5 — 정전 후 재개용. "
                         "분할은 시드 고정이라 결정론적)")
    args = ap.parse_args()
    only = (set(int(x) for x in args.only_folds.split(','))
            if args.only_folds else None)
    if args.epochs:
        T.EPOCHS = args.epochs
        print(f"!! 에폭 {args.epochs}로 축소 (예비 실험 모드)")

    _init_env()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    (flux, feat, labels, teff_n, logg_n, logg_ok, lum_idx, lum_ok,
     df, lt_sd, lg_sd, f_mu, f_sd) = load_pool()
    lum_counts = np.array([(df['LUM'] == c).sum() for c in LUM_ORDER])
    lum_w = effective_num_weights(lum_counts)
    groups = df["GROUP"].astype(str).values
    is_orig = (df["AUG"] == 0).values
    print(f"전체 풀: {len(flux):,}개 (원본 {int(is_orig.sum()):,}), "
          f"그룹 {df['GROUP'].nunique():,}개, {args.k}-fold, "
          f"epochs={T.EPOCHS}, device={DEVICE}")

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.k)

    feat_std_all = ((feat - f_mu) / f_sd).astype(np.float32)

    rows, lines = [], []
    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(flux, labels, groups), 1):
        if only is not None and fold not in only:
            continue
        va_idx = va_idx[is_orig[va_idx]]
        print("\n" + "=" * 60)
        print(f"[폴드 {fold}/{args.k}] train {len(tr_idx):,} / "
              f"val(원본) {len(va_idx):,}")
        assert not (set(groups[tr_idx]) & set(groups[va_idx]))

        cw = effective_num_weights(
            np.bincount(labels[tr_idx], minlength=N_CLASSES))
        sampler = WeightedRandomSampler(
            torch.from_numpy(cw[labels[tr_idx]]).float(),
            num_samples=len(tr_idx), replacement=True)
        ds_tr = SpecDatasetV5(flux, feat, tr_idx, labels, teff_n,
                              logg_n, logg_ok, lum_idx, lum_ok,
                              f_mu, f_sd, augment=True)
        ds_va = SpecDatasetV5(flux, feat, va_idx, labels, teff_n,
                              logg_n, logg_ok, lum_idx, lum_ok,
                              f_mu, f_sd, augment=False)
        l_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)
        l_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

        cnn = SpectralResNetV5().to(DEVICE)
        train_model(cnn, l_tr, l_va, f"f{fold}-res5",
                    os.path.join(TMP_DIR, "resnet.pth"),
                    teff_sd=lt_sd, logg_sd=lg_sd, lum_weights=lum_w)
        mlp = FeatureMLPV5().to(DEVICE)
        train_model(mlp, l_tr, l_va, f"f{fold}-mlp5",
                    os.path.join(TMP_DIR, "mlp.pth"),
                    teff_sd=lt_sd, logg_sd=lg_sd, lum_weights=lum_w)

        cnn.load_state_dict(torch.load(os.path.join(TMP_DIR, "resnet.pth"),
                                       map_location=DEVICE,
                                       weights_only=True))
        mlp.load_state_dict(torch.load(os.path.join(TMP_DIR, "mlp.pth"),
                                       map_location=DEVICE,
                                       weights_only=True))
        Pc = probs_of(cnn, flux[va_idx], feat_std_all[va_idx])
        Pm = probs_of(mlp, flux[va_idx], feat_std_all[va_idx])
        y = labels[va_idx]
        res = {"cnn": metrics(Pc.argmax(1), y),
               "mlp": metrics(Pm.argmax(1), y),
               "ens": metrics(((Pc + Pm) / 2).argmax(1), y)}
        rows.append(res)
        msg = f"[폴드 {fold}] " + "  ".join(
            f"{k}: {v[0]*100:.2f}% (OB {v[2][0]*100:.0f}%)"
            for k, v in res.items())
        print(msg); lines.append(msg)
        del ds_tr, ds_va, l_tr, l_va, cnn, mlp
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    lines.append("")
    names = {"cnn": "ResNet", "mlp": "MLP", "ens": "앙상블"}
    for k in rows[0]:
        accs = np.array([r[k][0] for r in rows]) * 100
        mac = np.array([r[k][1] for r in rows]) * 100
        obr = np.array([r[k][2][0] for r in rows]) * 100
        msg = (f"{names[k]:8s}: 정확도 {accs.mean():.2f}% ± {accs.std():.2f}%"
               f"   macro재현 {mac.mean():.2f}% ± {mac.std():.2f}%"
               f"   OB재현 {obr.mean():.1f}% ± {obr.std():.1f}%")
        print(msg); lines.append(msg)

    with open(os.path.join(OUT_DIR, "결과.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"v5 {args.k}-fold (별 그룹 단위, epochs={T.EPOCHS}, "
                f"6클래스, 39피처, 3-태스크)\n")
        f.write("\n".join(lines))
    print(f"\n저장: {os.path.join(OUT_DIR, '결과.txt')}")


if __name__ == "__main__":
    main()
