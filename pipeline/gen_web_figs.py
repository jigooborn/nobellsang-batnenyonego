# -*- coding: utf-8 -*-
"""웹사이트용 다크 테마 그림 — 배경 투명, 흰 글씨 (JIGOOBORN 사이트 톤)"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocess_core import CLASS_ORDER, N_CLASSES, FEATURE_NAMES, LINES_V5

OUT = r'C:\Users\user\Desktop\최종\github_repo\docs\figures'
os.makedirs(OUT, exist_ok=True)

DIM = '#8b8f98'
FG = '#e8eaee'
plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'figure.facecolor': 'none', 'axes.facecolor': 'none',
    'savefig.facecolor': 'none', 'savefig.transparent': True,
    'text.color': FG, 'axes.labelcolor': FG,
    'xtick.color': DIM, 'ytick.color': DIM,
    'axes.edgecolor': '#3a3f48', 'grid.color': '#2a2e35',
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.dpi': 130,
})

CLASS_C = {'OB': '#7fa8ff', 'A': '#a9c0e8', 'F': '#f0dd7a',
           'G': '#f5b731', 'K': '#e8834a', 'M': '#e05545'}


def finish(fig, name):
    fig.savefig(os.path.join(OUT, name), transparent=True, bbox_inches='tight',
                pad_inches=0.18)
    plt.close(fig)
    print(' ', name)


pred = pd.read_csv('results/eval_v5/test_predictions.csv')
lab = pd.read_csv('data/v5_test_labels.csv')
feat = np.load('data/v5_test_features.npy')
fi = {n: i for i, n in enumerate(FEATURE_NAMES)}

# ── 1. 대표 스펙트럼 3종 (뜨거움 → 차가움) ────────────────────────────
flux = np.load('data/v5_test_flux.npy', mmap_mode='r')
wave = np.linspace(4000, 7400, 3401)


def pick(cls):
    """해당 분광형에서 신뢰도 높고 깨끗한 스펙트럼 하나"""
    m = ((pred['cls_true'] == cls) & (pred['cls_pred'] == cls)
         & (pred['prob_ens'] > 0.97)).values
    idx = np.where(m)[0]
    best = None
    for i in idx[:400]:
        y = np.asarray(flux[i], dtype=np.float32)
        if not np.isfinite(y).all():
            continue
        if y.max() > 2.2 or y.min() < 0.05:
            continue
        if (y == 1.0).mean() > 0.012:
            continue
        score = np.nanstd(np.diff(y))          # 잡음이 적은 쪽
        if best is None or score < best[0]:
            best = (score, i, y)
    return best


SHOW = [('OB', '헬륨 흡수선이 살아 있는 고온성'),
        ('G', '태양과 같은 부류 — G밴드와 Mg b가 뚜렷'),
        ('M', 'TiO 분자띠가 만드는 톱니 무늬')]

fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)
for ax, (cls, note) in zip(axes, SHOW):
    got = pick(cls)
    if got is None:
        continue
    _, i, y = got
    ax.plot(wave, y, lw=0.85, color=CLASS_C[cls])
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    ax.set_ylim(lo - .05 * (hi - lo), hi + .34 * (hi - lo))  # 위쪽에 주석 자리
    r = pred.iloc[i]
    ax.text(0.014, 0.93, f"{cls}형", transform=ax.transAxes,
            fontsize=20, fontweight='bold', color=CLASS_C[cls], va='top')
    ax.text(0.986, 0.92, f"{r['teff_pred']:,.0f} K · {note}",
            transform=ax.transAxes, fontsize=12, color=DIM, va='top', ha='right')
    ax.set_ylabel('정규화 flux')
    ax.grid(alpha=0.22, lw=0.6)
    for s in ax.spines.values():
        s.set_alpha(0.5)
axes[-1].set_xlabel('파장 (Å)')
fig.tight_layout()
finish(fig, 'w_spectra.png')

# ── 2. 사하-볼츠만 곡선 (눈금 겹침 수정) ──────────────────────────────
teff = lab['TEFF'].values
show = [('ew_Hbeta', 'Hβ 수소선', '#7fa8ff'),
        ('ew_He4471', 'He I 헬륨선', '#a98bd8'),
        ('ew_Ca4227', 'Ca I 칼슘선', '#54c7ac'),
        ('ew_Mgb', 'Mg b 마그네슘', '#f5a04a'),
        ('ew_TiO2', 'TiO 분자띠', '#e05545')]
bins = np.logspace(np.log10(2800), np.log10(35000), 24)
fig, ax = plt.subplots(figsize=(12, 6.6))
for key, name, color in show:
    v = feat[:, fi[key]]
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (teff >= bins[i]) & (teff < bins[i + 1])
        if m.sum() >= 15:
            xs.append(np.sqrt(bins[i] * bins[i + 1]))
            ys.append(np.median(v[m]))
    ax.plot(xs, ys, 'o-', color=color, lw=2.8, ms=6, label=name)
ax.set_xscale('log')
ax.set_xlim(36000, 2700)
ticks = [30000, 20000, 10000, 7000, 5000, 3500]
ax.set_xticks(ticks)
ax.set_xticklabels([f'{t:,}' for t in ticks])
ax.set_xticks([], minor=True)                      # 겹치던 보조 눈금 제거
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FixedFormatter(
    [f'{t:,}' for t in ticks]))
for b in [10000, 7500, 6000, 5000, 3500]:
    ax.axvline(b, color='#3a3f48', ls=':', lw=1)
ymax = ax.get_ylim()[1]
for c, xm in [('OB', 19000), ('A', 8700), ('F', 6700), ('G', 5480),
              ('K', 4180), ('M', 3100)]:
    ax.text(xm, ymax * 0.97, c, fontsize=17, fontweight='bold',
            color=CLASS_C[c], alpha=.85, ha='center', va='top')
ax.set_xlabel('표면온도 (K)   ← 높음')
ax.set_ylabel('흡수선 세기 (등가폭, Å)')
leg = ax.legend(loc='upper left', facecolor='#12151a', edgecolor='#3a3f48',
                labelcolor=FG, framealpha=.85)
ax.grid(alpha=0.22, lw=0.6)
for s in ax.spines.values():
    s.set_alpha(0.5)
fig.tight_layout()
finish(fig, 'w_saha.png')

# ── 3. AI 예측만으로 그린 H-R도 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.4, 7.4))
for c in CLASS_ORDER:
    m = pred['cls_pred'] == c
    ax.scatter(pred.loc[m, 'teff_pred'], pred.loc[m, 'logg_pred'],
               s=5, alpha=0.38, c=CLASS_C[c], label=f'{c}형',
               edgecolors='none', rasterized=True)
ax.set_xscale('log')
ax.set_xlim(45000, 2700)
ax.set_ylim(5.6, 0)
ax.set_xticks([30000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(['30,000', '10,000', '6,000', '4,000', '3,000'])
ax.set_xticks([], minor=True)
ax.set_xlabel('AI 예측 표면온도 (K)   ← 높음')
ax.set_ylabel('AI 예측 표면중력 logg')
ax.annotate('주계열 (왜성)', xy=(6300, 4.55), fontsize=15, fontweight='bold',
            color='#ffffff', ha='center')
ax.annotate('적색거성가지', xy=(4150, 1.85), fontsize=15, fontweight='bold',
            color='#ffb38a', ha='center')
leg = ax.legend(markerscale=5, loc='lower left', ncol=2, facecolor='#12151a',
                edgecolor='#3a3f48', labelcolor=FG, framealpha=.85)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax.grid(alpha=0.18, which='major', lw=0.6)
for s in ax.spines.values():
    s.set_alpha(0.5)
fig.tight_layout()
finish(fig, 'w_hr.png')

# ── 4. 혼동행렬 ────────────────────────────────────────────────────
idx = {c: i for i, c in enumerate(CLASS_ORDER)}
y = pred['cls_true'].map(idx).values
p = pred['cls_pred'].map(idx).values
cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for a, b in zip(y, p):
    cm[a, b] += 1
norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
acc, adj = (y == p).mean(), (np.abs(y - p) <= 1).mean()

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('nb', ['#0d1016', '#1b3b6f', '#5b8ede', '#cfe0ff'])
fig, ax = plt.subplots(figsize=(8.2, 7.4))
ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if cm[i, j] == 0:
            continue
        ax.text(j, i, f'{cm[i, j]:,}\n{100*norm[i, j]:.1f}%', ha='center',
                va='center', fontsize=11.5,
                fontweight='bold' if i == j else 'normal',
                color='#0b0e13' if norm[i, j] > 0.55 else FG)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_ORDER, fontsize=15, color=FG)
ax.set_yticks(range(N_CLASSES))
ax.set_yticklabels([f'{c}  ({100*norm[k, k]:.0f}%)'
                    for k, c in enumerate(CLASS_ORDER)], fontsize=12.5, color=FG)
ax.set_xlabel('AI 예측')
ax.set_ylabel('정답 (재현율)')
print(f'   [cm] acc={100*acc:.2f}%  adj={100*adj:.2f}%  n={len(pred):,}')
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
finish(fig, 'w_cm.png')

# ── 5. 금속함량 축퇴 — 나아갈 방향 근거 ─────────────────────────────
feh = lab['FEH'].values
wrong = (y != p)
edges = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.6]
names = ['< -1.5', '-1.5 ~ -1.0', '-1.0 ~ -0.5', '-0.5 ~ 0.0', '> 0.0']
rates, ns = [], []
for i in range(len(edges) - 1):
    m = np.isfinite(feh) & (feh >= edges[i]) & (feh < edges[i + 1])
    rates.append(100 * wrong[m].mean() if m.sum() else 0)
    ns.append(int(m.sum()))
fig, ax = plt.subplots(figsize=(11, 5.6))
cols = ['#e05545', '#e8834a', '#f5b731', '#7fa8ff', '#5b8ede']
bars = ax.bar(names, rates, color=cols, width=.62, alpha=.9)
for b, r, n in zip(bars, rates, ns):
    ax.text(b.get_x() + b.get_width() / 2, r + max(rates) * .04,
            f'{r:.1f}%\nN={n:,}', ha='center', fontsize=12, color=FG)
ax.set_ylim(0, max(rates) * 1.32)
ax.set_xlabel('금속함량 [Fe/H]   ← 금속이 적은 별')
ax.set_ylabel('오분류율 (%)')
print('   [feh] ' + ' | '.join(f'{n}: {r:.1f}% (N={c:,})'
                                for n, r, c in zip(names, rates, ns)))
ax.grid(axis='y', alpha=0.22, lw=0.6)
for s in ax.spines.values():
    s.set_alpha(0.5)
fig.tight_layout()
finish(fig, 'w_feh.png')

# ── 6. 5-fold 교차검증 ─────────────────────────────────────────────
data = {'CNN (ResNet)': [97.92, 97.92, 97.71, 97.78, 97.68],
        'MLP (물리지표)': [96.91, 96.64, 96.80, 96.76, 96.68],
        '앙상블': [97.82, 97.73, 97.64, 97.73, 97.71]}
fig, ax = plt.subplots(figsize=(10, 5.8))
cols = ['#5b8ede', '#f5a04a', '#e0685a']
rng = np.random.default_rng(1)
for i, ((name, v), col) in enumerate(zip(data.items(), cols), 1):
    v = np.array(v)
    bp = ax.boxplot(v, positions=[i], widths=0.44, patch_artist=True,
                    whis=(0, 100), showfliers=False,
                    medianprops=dict(color=FG, lw=1.5),
                    whiskerprops=dict(color='#5a5f6a'),
                    capprops=dict(color='#5a5f6a'),
                    boxprops=dict(edgecolor='#5a5f6a'))
    bp['boxes'][0].set_facecolor(col)
    bp['boxes'][0].set_alpha(.55)
    ax.scatter(i + rng.uniform(-.08, .08, len(v)), v, s=34, color=FG,
               zorder=5, alpha=.9)
    ax.text(i, v.max() + .10, f'{v.mean():.2f}%  ±{v.std():.2f}',
            ha='center', fontsize=13, fontweight='bold', color=FG)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(list(data.keys()), fontsize=13, color=FG)
ax.set_ylabel('정확도 (%)')
ax.set_ylim(96.2, 98.4)
ax.grid(axis='y', alpha=0.22, lw=0.6)
for s in ax.spines.values():
    s.set_alpha(0.5)
fig.tight_layout()
finish(fig, 'w_kfold.png')

print('완료 →', OUT)
