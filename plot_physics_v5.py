# -*- coding: utf-8 -*-
"""
plot_physics_v5.py — v5.1 성능·물리 의미 종합 그래프 (보고서용)

입력: results/eval_v5/test_predictions.csv (eval_v5.py 산출)
      data/v5_test_flux.npy, v5_test_features.npy, v5_test_labels.csv
출력 (results/figures_v51/):
  1_reliability.png   신뢰도 보정 곡선 — "97%라고 말하면 실제 97% 맞나"
  2_hr_diagram.png    AI 예측만으로 그린 H-R도 (물리 구조 재현 증거)
  3_saha_curves.png   흡수선 세기-온도 곡선 (사하-볼츠만 법칙 재현)
  4_exemplars.png     분광형별 대표 스펙트럼 + 원소선 (자문 2-4 그림)
실행: python plot_physics_v5.py   (eval_v5.py 완료 후)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess_core import (CLASS_ORDER, FEATURE_NAMES, TARGET_WAVE,
                             LINES_V5, LUM_KO)

PRED = os.path.join('results', 'eval_v5', 'test_predictions.csv')
OUT = os.path.join('results', 'figures_v51')
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass

CLASS_C = {'OB': '#7b9fe0', 'A': '#a8b8d8', 'F': '#e8d76f',
           'G': '#f2b727', 'K': '#e07b39', 'M': '#c73e2e'}


def fig_reliability(df):
    """신뢰도 보정: 앙상블 확률 구간별 실제 정확도 + 확률 분포."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bins = np.arange(0.3, 1.0001, 0.05)
    centers, accs, cnts = [], [], []
    correct = (df['cls_pred'] == df['cls_true']).values
    p = df['prob_ens'].values
    for i in range(len(bins) - 1):
        m = (p >= bins[i]) & (p < bins[i + 1] + (i == len(bins) - 2))
        if m.sum() >= 20:
            centers.append((bins[i] + bins[i + 1]) / 2)
            accs.append(correct[m].mean())
            cnts.append(m.sum())
    ax.plot([0.3, 1], [0.3, 1], 'k--', lw=1, label='완벽한 보정선')
    ax.plot(centers, accs, 'o-', color='#c0392b', lw=2, ms=7,
            label='v5.1 실측')
    ax.set_xlabel('모델이 말한 신뢰도 (앙상블 확률)')
    ax.set_ylabel('실제 정답률')
    ax.set_title('신뢰도 보정 곡선 — 확률의 정직성 검증', fontsize=11.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.hist(p[correct], bins=30, alpha=0.75, color='#5b8db8',
            label=f'정답 (중앙 신뢰도 {np.median(p[correct])*100:.0f}%)')
    ax.hist(p[~correct], bins=30, alpha=0.75, color='#c0392b',
            label=f'오답 (중앙 {np.median(p[~correct])*100:.0f}%)')
    ax.set_yscale('log')
    ax.set_xlabel('앙상블 확률')
    ax.set_ylabel('별 수 (로그)')
    ax.set_title('정답/오답별 신뢰도 분포 — 오답은 스스로 낮은 확률', fontsize=11.5)
    ax.legend(fontsize=9.5)
    fig.suptitle(f"v5.1 신뢰도 검증 (test {len(df):,}개)",
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '1_reliability.png'), dpi=150,
                bbox_inches='tight')
    print('1_reliability.png')


def fig_hr(df):
    """AI 예측만으로 그린 H-R도 — 주계열·거성가지 재현 여부."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for c in CLASS_ORDER:
        m = df['cls_pred'] == c
        ax.scatter(df.loc[m, 'teff_pred'], df.loc[m, 'logg_pred'],
                   s=4, alpha=0.35, c=CLASS_C[c], label=f'{c}형',
                   edgecolors='none', rasterized=True)
    ax.set_xscale('log')
    ax.set_xlim(45000, 2700)          # 천문학 관례: 온도 왼쪽이 높음
    ax.set_ylim(5.6, 0)               # logg 위가 작음(거성 위)
    ax.set_xlabel('AI 예측 유효온도 Teff (K) ← 높음')
    ax.set_ylabel('AI 예측 표면중력 logg (dex)')
    ax.annotate('주계열\n(왜성)', xy=(6000, 4.4), fontsize=12,
                fontweight='bold', color='#333', ha='center')
    ax.annotate('적색거성가지', xy=(4300, 2.2), fontsize=12,
                fontweight='bold', color='#8e1e12', ha='center')
    leg = ax.legend(fontsize=10, markerscale=4, loc='lower left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_title('AI 예측만으로 그린 H-R도 (킬-왜성 다이어그램)\n'
                 '— 스펙트럼에서 별의 진화 단계 구조가 복원됨',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '2_hr_diagram.png'), dpi=150,
                bbox_inches='tight')
    print('2_hr_diagram.png')


def fig_saha():
    """흡수선 등가폭-온도 곡선: 사하-볼츠만 법칙의 데이터 재현."""
    feat = np.load('data/v5_test_features.npy')
    lab = pd.read_csv('data/v5_test_labels.csv')
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    teff = lab['TEFF'].values
    show = [('ew_Hbeta', 'Hβ (수소 발머선)', '#5b8db8'),
            ('ew_He4471', 'He I 4471', '#7b52ab'),
            ('ew_Ca4227', 'Ca I 4227', '#2a9d8f'),
            ('ew_Mgb', 'Mg b 삼중선', '#e07b39'),
            ('ew_Fe4383', 'Fe I 4383', '#888'),
            ('ew_TiO2', 'TiO₂ 분자띠', '#c73e2e')]
    bins = np.logspace(np.log10(2800), np.log10(35000), 26)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for key, name, color in show:
        v = feat[:, fi[key]]
        xs, ys = [], []
        for i in range(len(bins) - 1):
            m = (teff >= bins[i]) & (teff < bins[i + 1])
            if m.sum() >= 15:
                xs.append(np.sqrt(bins[i] * bins[i + 1]))
                ys.append(np.median(v[m]))
        ax.plot(xs, ys, 'o-', color=color, lw=2, ms=5, label=name)
    ax.set_xscale('log')
    ax.set_xlim(36000, 2700)
    ax.set_xticks([30000, 20000, 10000, 7000, 5000, 3500])
    ax.set_xticklabels(['30,000', '20,000', '10,000', '7,000',
                        '5,000', '3,500'])
    for b, c in zip([10000, 7500, 6000, 5000, 3500],
                    ['OB|A', 'A|F', 'F|G', 'G|K', 'K|M']):
        ax.axvline(b, color='#bbb', ls=':', lw=0.8)
        ax.text(b, ax.get_ylim()[1], f' {c}', fontsize=8, color='#999',
                va='top')
    ax.set_xlabel('유효온도 Teff (K) ← 높음')
    ax.set_ylabel('흡수선 등가폭 (Å, test 표본 중앙값)')
    ax.set_title('흡수선 세기 - 온도 곡선 (test 표본 실측)\n'
                 '— 사하-볼츠만 법칙: 각 원소선은 자기 온도에서만 최대가 된다',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '3_saha_curves.png'), dpi=150,
                bbox_inches='tight')
    print('3_saha_curves.png')


def fig_exemplars(df):
    """분광형별 대표 스펙트럼 6장 + 원소선 라벨 (자문 피드백 2-4)."""
    from classify_gui_v5 import draw_spectrum, LINE_SENS
    flux = np.load('data/v5_test_flux.npy', mmap_mode='r')
    lab = pd.read_csv('data/v5_test_labels.csv')
    feat = np.load('data/v5_test_features.npy')
    n_lines = len(LINES_V5)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    for ax, c in zip(axes.flat, CLASS_ORDER):
        # 대표 선정: 정답+고신뢰 별 중 SNR 좋은 것 (pred csv와 행 정렬 동일)
        m = (df['cls_true'] == c) & (df['cls_pred'] == c) & \
            (df['prob_ens'] > 0.95)
        idx = df[m].index
        if not len(idx):
            idx = df[df['cls_true'] == c].index
        k = int(idx[0])
        ews = dict(zip([l[0] for l in LINES_V5],
                       feat[k, :n_lines].tolist()))
        fake_result = {'flux_norm': np.array(flux[k]), 'pred_ens': c,
                       'ews': ews}
        draw_spectrum(ax, fake_result, '', show_lines=True,
                      highlight=True)
        t = df.loc[k, 'teff_pred']
        lum = LUM_KO.get(df.loc[k, 'lum_pred'], '')
        ax.set_title(f"{c}형 대표 스펙트럼 (AI: 약 {t:,.0f} K · {lum})"
                     f" — 판정 근거 선은 붉은색",
                     fontsize=11, fontweight='bold')
    fig.suptitle('분광형별 대표 스펙트럼과 판정 근거 원소선 (test 실측)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '4_exemplars.png'), dpi=140,
                bbox_inches='tight')
    print('4_exemplars.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(PRED)
    fig_reliability(df)
    fig_hr(df)
    fig_saha()
    fig_exemplars(df)
    print(f"완료 → {OUT}/")


if __name__ == '__main__':
    main()
