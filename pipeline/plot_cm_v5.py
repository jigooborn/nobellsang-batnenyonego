# -*- coding: utf-8 -*-
"""
plot_cm_v5.py — 혼동행렬 그림 재생성 (정확도 표기 포함)

eval_all_v5.py 가 저장한 predictions_all_v5.csv 를 읽어 GPU 재추론 없이
그림만 다시 그림. 각 칸에 개수+행 비율(%), 대각선에 클래스 재현율,
제목에 전체 정확도·±1등급 표기.

실행: python plot_cm_v5.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess_core import CLASS_ORDER, N_CLASSES

PRED = os.path.join("results", "eval_all_v5", "predictions_all_v5.csv")
OUT = os.path.join("results", "eval_all_v5")

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass


def draw_cm(ax, sub, title):
    y, p = sub['label'].values, sub['pred'].values
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    acc = (y == p).mean()
    adj = (np.abs(y - p) <= 1).mean()
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] == 0:
                continue
            txt = f"{cm[i, j]:,}\n{100*norm[i, j]:.1f}%"
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5,
                    fontweight='bold' if i == j else 'normal',
                    color='white' if norm[i, j] > 0.5 else '#333')
    rec = [f"{c}\n({100*norm[k, k]:.0f}%)"
           for k, c in enumerate(CLASS_ORDER)]
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(rec, fontsize=9)
    ax.set_xlabel("AI 예측"); ax.set_ylabel("정답 (재현율)")
    ax.set_title(f"{title}\n정확도 {100*acc:.2f}% · ±1등급 {100*adj:.2f}%"
                 f" · N={len(sub):,}", fontsize=11)


def main():
    allr = pd.read_csv(PRED)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    draw_cm(axes[0], allr[allr['member'] == 'test'],
            "test (공식 성능 — 학습에 안 쓴 별)")
    draw_cm(axes[1], allr, "전체 24만 개 (train 포함 — 참고용)")
    fig.suptitle("v5 앙상블 혼동행렬 — 6클래스 (OB/A/F/G/K/M)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT, "혼동행렬_정확도표기.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"저장: {out}")

    # 서베이별 4분할도 추가
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    for ax, sv in zip(axes.flat, ['lamost', 'mastar', 'segue1', 'miles']):
        draw_cm(ax, allr[allr['survey'] == sv], f"서베이: {sv}")
    fig.suptitle("v5 서베이별 혼동행렬 (편중 진단용)", fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    out2 = os.path.join(OUT, "혼동행렬_서베이별.png")
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"저장: {out2}")


if __name__ == '__main__':
    main()
