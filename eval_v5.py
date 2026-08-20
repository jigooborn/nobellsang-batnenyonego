# -*- coding: utf-8 -*-
"""
eval_v5.py — v5 최종 평가 (test셋)

출력:
  1) CNN / MLP / 앙상블 정확도, ±1등급, 클래스별 재현율, 혼동행렬
  2) Teff 중앙 상대오차, logg MAE
  3) 광도계급 3분류 (전용 분류 헤드: 거성/주계열/백색왜성)
  4) 상관 플롯 (자문 피드백 2-2): 예측 Teff vs 실측, 예측 logg vs 실측
  5) 앙상블 물리 해석용: CNN 우세/MLP 우세 클래스 표 (피드백 2-3)

실행: python eval_v5.py   (train_v5.py 완료 후)
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess_core import (CLASS_ORDER, N_CLASSES, N_FEATURES,
                             LUM_ORDER, N_LUM, LUM_KO)
from train_v5 import SpectralResNetV5, FeatureMLPV5, DEVICE

DATA = "data"
OUT = os.path.join("results", "eval_v5")
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass


@torch.no_grad()
def infer(model, flux, feat, bs=512):
    model.eval()
    probs, teffs, loggs, lums = [], [], [], []
    for s in range(0, len(flux), bs):
        fb = torch.from_numpy(np.array(flux[s:s+bs], dtype=np.float32)
                              ).to(DEVICE)
        xb = torch.from_numpy(feat[s:s+bs]).to(DEVICE)
        lo, tp, gp, lu = model(fb, xb)
        probs.append(torch.softmax(lo, 1).cpu().numpy())
        teffs.append(tp.cpu().numpy())
        loggs.append(gp.cpu().numpy())
        lums.append(torch.softmax(lu, 1).cpu().numpy())
    return (np.concatenate(probs), np.concatenate(teffs),
            np.concatenate(loggs), np.concatenate(lums))


def report(name, pred, y, f):
    acc = (pred == y).mean()
    adj = (np.abs(pred - y) <= 1).mean()
    rec = []
    for k in range(N_CLASSES):
        m = y == k
        rec.append(100 * (pred[m] == k).mean() if m.sum() else np.nan)
    line = (f"[{name:9s}] {100*acc:.2f}%  ±1등급 {100*adj:.2f}%  " +
            " ".join(f"{c}:{r:.0f}" for c, r in zip(CLASS_ORDER, rec)))
    print(line); f.write(line + "\n")
    return acc, rec


def main():
    os.makedirs(OUT, exist_ok=True)
    flux = np.load(f"{DATA}/v5_test_flux.npy", mmap_mode='r')
    feat_raw = np.load(f"{DATA}/v5_test_features.npy")
    df = pd.read_csv(f"{DATA}/v5_test_labels.csv")
    y = np.array([CLASS_ORDER.index(l) for l in df['LABEL']])

    fstats = np.load(f"{DATA}/feat_norm_v5.npy")
    feat = ((feat_raw - fstats[0]) / fstats[1]).astype(np.float32)
    lt_mu, lt_sd = np.load(f"{DATA}/teff_norm_v5.npy")
    lg_mu, lg_sd = np.load(f"{DATA}/logg_norm_v5.npy")

    cnn = SpectralResNetV5().to(DEVICE)
    cnn.load_state_dict(torch.load("models/resnet_v5.pth",
                                   map_location=DEVICE))
    mlp = FeatureMLPV5().to(DEVICE)
    mlp.load_state_dict(torch.load("models/mlp_v5.pth",
                                   map_location=DEVICE))

    print(f"test {len(df):,}개 추론 중...")
    p_c, t_c, g_c, u_c = infer(cnn, flux, feat)
    p_m, t_m, g_m, u_m = infer(mlp, flux, feat)
    p_e = (p_c + p_m) / 2
    u_e = (u_c + u_m) / 2

    f = open(os.path.join(OUT, "eval_v5.txt"), 'w', encoding='utf-8')
    f.write(f"v5 최종 평가 — test {len(df):,}개, 6클래스(교과서 경계)\n\n")

    pred_c, pred_m, pred_e = p_c.argmax(1), p_m.argmax(1), p_e.argmax(1)
    report("ResNet", pred_c, y, f)
    report("MLP", pred_m, y, f)
    acc_e, _ = report("Ensemble", pred_e, y, f)

    # 혼동행렬 (앙상블)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for a, b in zip(y, pred_e):
        cm[a, b] += 1
    f.write("\n혼동행렬 (행=실제, 열=예측):\n")
    f.write("       " + "".join(f"{c:>8s}" for c in CLASS_ORDER) + "\n")
    for k in range(N_CLASSES):
        f.write(f"{CLASS_ORDER[k]:>5s}  " +
                "".join(f"{cm[k, j]:8d}" for j in range(N_CLASSES)) + "\n")

    # CNN vs MLP 우세 구간 (피드백 2-3 — 앙상블 물리 해석)
    f.write("\n[CNN vs MLP 클래스별 재현율 — 앙상블 해석용]\n")
    print("\n[CNN vs MLP 우세 구간]")
    for k, c in enumerate(CLASS_ORDER):
        m = y == k
        rc = 100 * (pred_c[m] == k).mean()
        rm = 100 * (pred_m[m] == k).mean()
        who = 'CNN' if rc > rm + 0.5 else ('MLP' if rm > rc + 0.5 else '동률')
        line = f"  {c}: CNN {rc:.1f}% vs MLP {rm:.1f}%  → {who}"
        print(line); f.write(line + "\n")

    # Teff 회귀 (앙상블 = 두 모델 평균)
    teff_pred = 10 ** (((t_c + t_m) / 2) * lt_sd + lt_mu)
    teff_true = df['TEFF'].values
    rel = np.abs(teff_pred - teff_true) / teff_true
    line = f"\n[Teff] 중앙 상대오차 {100*np.median(rel):.2f}%"
    print(line); f.write(line + "\n")

    # logg 회귀 + 광도계급
    logg_pred = ((g_c + g_m) / 2) * lg_sd + lg_mu
    ok = np.isfinite(df['LOGG'].values)
    mae = np.abs(logg_pred[ok] - df['LOGG'].values[ok])
    line = (f"[logg] MAE {mae.mean():.3f} dex (중앙 {np.median(mae):.3f})")
    print(line); f.write(line + "\n")

    lum_true = df['LUM'].values
    lum_pred = np.array([LUM_ORDER[i] for i in u_e.argmax(1)])
    known = np.isin(lum_true, LUM_ORDER)
    acc_lum = (lum_pred[known] == lum_true[known]).mean()
    line = (f"[광도계급 3분류 — 전용 헤드] {100*acc_lum:.2f}% "
            f"(N={known.sum():,})")
    print(line); f.write(line + "\n")
    for lum in LUM_ORDER:
        m = lum_true == lum
        if m.sum():
            r = 100 * (lum_pred[m] == lum).mean()
            line = f"  {LUM_KO[lum]}: {r:.1f}% (N={m.sum():,})"
            print(line); f.write(line + "\n")

    # ── 상관 플롯 (피드백 2-2) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(teff_true, teff_pred, s=2, alpha=0.15, c='#1f77b4',
               rasterized=True)
    lim = [2500, 60000]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('실측 Teff (K)'); ax.set_ylabel('AI 예측 Teff (K)')
    ax.set_title(f'유효온도: 예측 vs 실측 (중앙 오차 {100*np.median(rel):.2f}%)')
    ax = axes[1]
    ax.scatter(df['LOGG'].values[ok], logg_pred[ok], s=2, alpha=0.15,
               c='#2ca02c', rasterized=True)
    ax.plot([0, 8], [0, 8], 'r--', lw=1)
    ax.axhline(3.5, color='gray', ls=':', lw=0.8)
    ax.axhline(5.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('실측 logg (dex)'); ax.set_ylabel('AI 예측 logg (dex)')
    ax.set_title(f'표면중력: 예측 vs 실측 (MAE {mae.mean():.3f} dex)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_plots.png"), dpi=150,
                bbox_inches='tight')
    print(f"\n상관 플롯 저장: {OUT}/correlation_plots.png")

    # 별별 예측 전체 저장 (H-R도·신뢰도 곡선 등 후속 분석용)
    pd.DataFrame({
        'star_id': df['STAR_ID'], 'source': df['SOURCE'],
        'teff_true': teff_true, 'logg_true': df['LOGG'].values,
        'cls_true': df['LABEL'], 'lum_true': lum_true,
        'cls_pred': [CLASS_ORDER[i] for i in pred_e],
        'prob_ens': p_e.max(1),
        'teff_pred': teff_pred, 'logg_pred': logg_pred,
        'lum_pred': lum_pred,
    }).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
    f.close()
    print(f"보고서: {OUT}/eval_v5.txt (+ test_predictions.csv)")


if __name__ == "__main__":
    main()
