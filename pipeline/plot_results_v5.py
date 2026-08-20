# -*- coding: utf-8 -*-
"""
plot_results_v5.py — v5 검증 결과 종합 그래프 (보고서용)

출력 (results/figures/):
  1_kfold.png       5-fold 안정성 (모델별 폴드 정확도 + 평균±표준편차)
  2_xsl.png         XSL 외부 검증 (클래스별 재현율 + Teff 산점도)
  3_summary.png     검증 4종 종합 비교 (+ v4 기준선)
  4_simbad.png      SIMBAD 재검증 (일치율 + 불일치 심판)
실행: python plot_results_v5.py
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess_core import CLASS_ORDER, teff_to_class

OUT = os.path.join('results', 'figures')
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass

C_CNN, C_MLP, C_ENS = '#8fb8de', '#f0b27a', '#c0392b'


def fig_kfold():
    txt = open(os.path.join('results', 'kfold_v5', '결과.txt'),
               encoding='utf-8').read()
    folds = {'cnn': [], 'mlp': [], 'ens': []}
    for m in re.finditer(
            r'cnn: ([\d.]+)%.*?mlp: ([\d.]+)%.*?ens: ([\d.]+)%', txt):
        folds['cnn'].append(float(m.group(1)))
        folds['mlp'].append(float(m.group(2)))
        folds['ens'].append(float(m.group(3)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, len(folds['ens']) + 1)
    for key, c, name in [('cnn', C_CNN, 'CNN(ResNet)'),
                         ('mlp', C_MLP, 'MLP(흡수선 피처)'),
                         ('ens', C_ENS, '앙상블')]:
        v = np.array(folds[key])
        lw = 2.5 if key == 'ens' else 1.5
        ax.plot(x, v, 'o-', color=c, lw=lw, ms=7, label=f"{name}  "
                f"{v.mean():.2f}±{v.std():.2f}%")
    ens = np.array(folds['ens'])
    ax.axhspan(ens.mean() - ens.std(), ens.mean() + ens.std(),
               color=C_ENS, alpha=0.08)
    ax.axhline(96.99, color='gray', ls='--', lw=1.2)
    ax.text(len(x) + 0.05, 96.99, 'v4 앙상블\n96.99±0.30%', fontsize=8.5,
            color='gray', va='center')
    ax.set_xticks(x)
    ax.set_xlabel('폴드 (별 그룹 단위 분할)')
    ax.set_ylabel('정확도 (%)')
    ax.set_ylim(95.8, 98.2)
    ax.set_xlim(0.6, len(x) + 0.9)
    ax.legend(loc='lower left', fontsize=9.5)
    ax.set_title('5-fold 교차검증 — 분할 방식과 무관하게 일관된 성능',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '1_kfold.png'), dpi=150,
                bbox_inches='tight')
    print('1_kfold.png')


def fig_xsl():
    df = pd.read_csv(os.path.join('results', 'xsl_external',
                                  'xsl_predictions.csv'))
    ok = df[df['cls_pred'].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    recs, ns = [], []
    for c in CLASS_ORDER:
        s = ok[ok['cls_true'] == c]
        recs.append(100 * (s['cls_pred'] == c).mean() if len(s) else 0)
        ns.append(len(s))
    bars = ax.bar(CLASS_ORDER, recs, color=C_ENS, alpha=0.85)
    for b, r, n in zip(bars, recs, ns):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%",
                ha='center', fontsize=10, fontweight='bold')
        ax.text(b.get_x() + b.get_width() / 2, 4, f"N={n}",
                ha='center', fontsize=8.5, color='white')
    acc = 100 * (ok['cls_pred'] == ok['cls_true']).mean()
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    adj = 100 * ((ok['cls_pred'].map(idx)
                  - ok['cls_true'].map(idx)).abs() <= 1).mean()
    ax.set_ylim(0, 108)
    ax.set_ylabel('재현율 (%)')
    ax.set_title(f"클래스별 재현율 — 전체 {acc:.1f}% · ±1등급 {adj:.1f}%",
                 fontsize=11)
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1]
    ax.scatter(ok['teff_true'], ok['teff_pred'], s=14, alpha=0.45,
               c='#1f77b4', edgecolors='none', rasterized=True)
    lim = [2500, 45000]
    ax.plot(lim, lim, 'r--', lw=1.2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    rel = (ok['teff_pred'] - ok['teff_true']).abs() / ok['teff_true']
    ax.set_xlabel('문헌 Teff (K) — Arentsen+2019, 독립 파이프라인')
    ax.set_ylabel('AI 예측 Teff (K)')
    ax.set_title(f"유효온도 예측 vs 실측 (중앙 오차 {100*rel.median():.1f}%)",
                 fontsize=11)
    ax.grid(alpha=0.25, which='both')
    fig.suptitle(f"XSL DR3 외부 검증 (VLT/X-shooter, {len(ok)}개 — "
                 "학습에 전혀 사용 안 된 망원경·기기)",
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '2_xsl.png'), dpi=150,
                bbox_inches='tight')
    print('2_xsl.png')


def fig_summary():
    rows = [
        ('test\n(14,471)', 97.52, 99.72),
        ('5-fold 평균\n(별그룹 분할)', 97.47, None),
        ('MILES 실전\n(985)', 94.8, 100.0),
        ('XSL 외부\n(751, VLT)', 91.1, 98.3),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(rows))
    accs = [r[1] for r in rows]
    adjs = [r[2] for r in rows]
    b1 = ax.bar(x - 0.2, accs, 0.38, color=C_ENS, alpha=0.9,
                label='정확도')
    b2 = ax.bar(x + 0.2, [a if a else 0 for a in adjs], 0.38,
                color='#5b8db8', alpha=0.9, label='±1등급 정확도')
    for b, v in zip(b1, accs):
        ax.text(b.get_x() + 0.19, v + 0.4, f"{v:.1f}", ha='center',
                fontsize=10.5, fontweight='bold', color=C_ENS)
    for b, v in zip(b2, adjs):
        if v:
            ax.text(b.get_x() + 0.19, v + 0.4, f"{v:.1f}", ha='center',
                    fontsize=10.5, fontweight='bold', color='#5b8db8')
    ax.errorbar([1 - 0.2], [97.47], yerr=[0.20], fmt='none',
                ecolor='k', capsize=5, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(85, 102.5)
    ax.set_ylabel('정확도 (%)')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('v5 검증 종합 — 내부에서 외부로 갈수록 엄격한 4단계 검증',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '3_summary.png'), dpi=150,
                bbox_inches='tight')
    print('3_summary.png')


def fig_simbad():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    vals = [356, 107, 6]     # 일치 / ±1 불일치 / 2단계+ 불일치 (469 기준)
    total = sum(vals)
    labels = [f'일치\n{vals[0]}개 ({100*vals[0]/total:.0f}%)',
              f'±1클래스 불일치\n{vals[1]}개 ({100*vals[1]/total:.0f}%)',
              f'2클래스 이상\n{vals[2]}개 ({100*vals[2]/total:.0f}%)']
    ax.pie(vals, labels=labels, colors=['#5b8db8', '#f0b27a', '#c0392b'],
           startangle=90, textprops={'fontsize': 10},
           wedgeprops={'edgecolor': 'white', 'lw': 1.5})
    ax.set_title(f'AI ↔ SIMBAD 등록 분광형 비교 (MILES {total}개)',
                 fontsize=11)

    ax = axes[1]
    cats = ['독립 문헌 Teff가\nAI 판정과 일치\n(SIMBAD 재검토 후보)',
            '문헌 Teff가\nSIMBAD 편\n(AI 오분류)', 'Teff가\n제3의 구간']
    vals2 = [93, 18, 2]
    bars = ax.bar(cats, vals2, color=['#c0392b', '#5b8db8', '#999'],
                  alpha=0.9)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}개",
                ha='center', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_ylabel('불일치 사례 수')
    ax.set_title('불일치 113개의 "심판" — 제3의 독립 근거(문헌 Teff)로 판정',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    fig.suptitle('SIMBAD 레거시 라벨 재검증 — 이질적 옛 분류의 독립 재분류',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '4_simbad.png'), dpi=150,
                bbox_inches='tight')
    print('4_simbad.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_kfold()
    fig_xsl()
    fig_summary()
    fig_simbad()
    print(f"완료 → {OUT}/")


if __name__ == '__main__':
    main()
