# -*- coding: utf-8 -*-
"""
gen_exemplars3.py — 분광형별 대표 스펙트럼 3개씩 (보고서 삽입용)

선정 기준: test 셋에서 정답 + 앙상블 신뢰도 ≥97% 인 별 중,
클래스 내 온도 하위/중간/상위에서 1개씩 (클래스 안의 다양성 표시).
각 장: 3단 스택, 27개 원소선 라벨 + 판정 근거 선 붉은 강조.
출력: results/figures_v51/exemplars/대표_<class>.png (6장)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess_core import CLASS_ORDER, LINES_V5, LUM_KO
from classify_gui_v5 import draw_spectrum

OUT = os.path.join('results', 'figures_v51', 'exemplars')
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass


def main():
    os.makedirs(OUT, exist_ok=True)
    pred = pd.read_csv(os.path.join('results', 'eval_v5',
                                    'test_predictions.csv'))
    flux = np.load('data/v5_test_flux.npy', mmap_mode='r')
    feat = np.load('data/v5_test_features.npy')
    n_lines = len(LINES_V5)

    for c in CLASS_ORDER:
        cand = pred[(pred['cls_true'] == c) & (pred['cls_pred'] == c)
                    & (pred['prob_ens'] >= 0.97)].copy()
        if len(cand) < 3:
            cand = pred[(pred['cls_true'] == c)
                        & (pred['cls_pred'] == c)].copy()
        cand = cand.sort_values('teff_true')

        def clean(k):
            """그림 품질 검사: 스파이크·붕괴 패딩 구간이 없는 스펙트럼만."""
            f = np.array(flux[int(k)])
            return (f.max() < 2.2 and f.min() > 0.05
                    and (f == 1.0).mean() < 0.03)

        # 온도 하/중/상 지점에서 시작해 깨끗한 스펙트럼이 나올 때까지 탐색
        seen, final = set(), []
        for q in (0.1, 0.5, 0.9):
            start = int(q * (len(cand) - 1))
            pick = None
            for off in range(len(cand)):
                for j in (start + off, start - off):
                    if 0 <= j < len(cand):
                        row = cand.iloc[j]
                        if row['star_id'] not in seen and clean(row.name):
                            pick = row
                            break
                if pick is not None:
                    break
            if pick is None:            # 전부 지저분하면 그냥 원래 지점
                pick = cand.iloc[start]
            seen.add(pick['star_id'])
            final.append(pick)

        fig, axes = plt.subplots(3, 1, figsize=(13, 12))
        for ax, row in zip(axes, final):
            k = int(row.name)
            ews = dict(zip([l[0] for l in LINES_V5],
                           feat[k, :n_lines].tolist()))
            res = {'flux_norm': np.array(flux[k]), 'pred_ens': c,
                   'ews': ews}
            draw_spectrum(ax, res, '', show_lines=True, highlight=True)
            lum = LUM_KO.get(row['lum_pred'], row['lum_pred'])
            ax.set_title(
                f"{row['star_id']}  ({row['source']})  —  "
                f"AI: {c}형 (약 {row['teff_pred']:,.0f} K) · {lum} · "
                f"신뢰도 {row['prob_ens']*100:.1f}%   "
                f"[실측 Teff {row['teff_true']:,.0f} K]",
                fontsize=10.5, fontweight='bold')
        fig.suptitle(f"{c}형 대표 스펙트럼 3선 — 판정 근거 원소선은 붉은색 "
                     f"(test 셋, 학습 미사용 별)",
                     fontsize=13.5, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(OUT, f'대표_{c}.png')
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(out, '←', [str(r['star_id'])[:24] for r in final])


if __name__ == '__main__':
    main()
