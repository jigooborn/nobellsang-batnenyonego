# -*- coding: utf-8 -*-
"""전처리 단계별 그림 10장 (원본 + 9단계) — 웹 인터랙티브용.

실제 프로그램이 샘플 스펙트럼에 적용하는 RV·A_V 를 그대로 사용해,
각 단계에서 무엇이 달라지는지 이전 단계와 겹쳐 보여준다.
모든 그림은 동일한 픽셀 크기로 저장되어 전환 시 흔들리지 않는다.
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

from preprocess_core import (vac_to_air, safe_arr, estimate_rv, cardelli_alav,
                             fit_continuum, kappa_sigma_clip, C_KMS,
                             WAVE_MIN, WAVE_MAX, TARGET_WAVE,
                             CONT_ITERS, SIGMA_CUT, KSIG_ITERS, SG_WIN,
                             CONT_FLOOR)

SRC = r'..\github_repo\samples\spec_401011246.fits'
OUT = r'C:\Users\user\Desktop\최종\github_repo\docs\figures'
os.makedirs(OUT, exist_ok=True)

DIM, FG, PREV = '#8b8f98', '#e8eaee', '#474d57'
CUR = '#f5b731'                       # G형 색
plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'figure.facecolor': 'none', 'axes.facecolor': 'none',
    'savefig.facecolor': 'none', 'text.color': FG, 'axes.labelcolor': DIM,
    'xtick.color': DIM, 'ytick.color': DIM,
    'axes.edgecolor': '#343943', 'grid.color': '#23272e',
    'font.size': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

# ── 샘플 읽기 (LAMOST) ────────────────────────────────────────────────
h = fits.open(SRC)
names = h[1].columns.names
low = [c.lower() for c in names]
d = h[1].data
flux_raw = safe_arr(d[names[low.index('flux')]]).flatten().astype(float)
wave_vac = safe_arr(d[names[low.index('wavelength')]]).flatten().astype(float)

# ── 프로그램이 실제로 쓰는 RV·A_V ──────────────────────────────────────
import sys
sys.path.insert(0, '.')
RV = AV = None
try:
    from classify_gui_v5 import PredictorV5
    P = PredictorV5(model_dir=r'..\github_repo\models')
    r = P.predict_auto(vac_to_air(wave_vac), flux_raw)
    RV, AV = r['rv_used'], r['av_used']
    print(f"실제 판정: {r['pred_ens']}형 {r['teff_ens']:,.0f} K "
          f"(신뢰도 {100*r['prob_ens']:.0f}%) · {r['lum']}  "
          f"RV={RV:.1f} km/s  A_V={AV}")
except Exception as e:
    print('모델 경로 실패 →', e)
    RV = estimate_rv(vac_to_air(wave_vac), flux_raw)
    AV = None
if RV is None:
    RV = 0.0

# ── 단계별 상태 계산 ──────────────────────────────────────────────────
stages = []          # (wave, flux, 종류) 종류: 'raw' | 'norm'

w = wave_vac.copy()
f = flux_raw.copy()
stages.append((w.copy(), f.copy(), 'raw'))                      # 0 원본

w = vac_to_air(w)
stages.append((w.copy(), f.copy(), 'raw'))                      # 1 진공→공기

good = np.isfinite(f) & (f > 0)
n_bad = int((~good).sum())
if not good.all():
    gi = np.where(good)[0]
    f = np.interp(np.arange(len(f)), gi, f[gi])
stages.append((w.copy(), f.copy(), 'raw'))                      # 2 불량화소

w = w / (1.0 + RV / C_KMS)
stages.append((w.copy(), f.copy(), 'raw'))                      # 3 도플러

if AV is not None and np.isfinite(AV) and AV > 0:
    f = f * 10.0 ** (0.4 * AV * cardelli_alav(w))
stages.append((w.copy(), f.copy(), 'raw'))                      # 4 소광

m = (w >= WAVE_MIN) & (w <= WAVE_MAX)
w, f = w[m], f[m]
o = np.argsort(w)
w, f = w[o], f[o]
stages.append((w.copy(), f.copy(), 'raw'))                      # 5 파장컷

wn = (w - WAVE_MIN) / (WAVE_MAX - WAVE_MIN) * 2.0 - 1.0
cont1 = fit_continuum(wn, f, deg=3, iters=CONT_ITERS)
f = kappa_sigma_clip(f, cont1, kappa=SIGMA_CUT, iters=KSIG_ITERS)
stages.append((w.copy(), f.copy(), 'raw'))                      # 6 이상값

win = min(SG_WIN, (len(f) // 30) * 2 + 1)
win = max(win, 5)
win = win + 1 if win % 2 == 0 else win
f = savgol_filter(f, window_length=win, polyorder=3)
stages.append((w.copy(), f.copy(), 'raw'))                      # 7 평활

cont2 = fit_continuum(wn, f, deg=3, iters=CONT_ITERS)
med = np.median(cont2)
lowc = cont2 < CONT_FLOOR * med
fn = f / np.where(lowc, med, cont2)
fn[lowc] = 1.0
fn[~np.isfinite(fn)] = 1.0
stages.append((w.copy(), fn.copy(), 'norm'))                    # 8 정규화

fo = np.interp(TARGET_WAVE, w, fn, left=1.0, right=1.0)
fo = np.clip(fo, 0.0, 5.0)
stages.append((TARGET_WAVE.copy(), fo.copy(), 'norm'))          # 9 격자

LABELS = [
    '원본 관측 스펙트럼',
    '01 · 진공 → 공기 파장 변환',
    '02 · 불량 화소 보간',
    '03 · 시선속도(도플러) 보정',
    '04 · 성간소광 보정',
    '05 · 파장 구간 절단',
    '06 · 이상값 제거',
    '07 · 평활화',
    '08 · 연속선 정규화',
    '09 · 공통 격자 정렬',
]

ZOOM = (4845.0, 4878.0)        # Hβ 4861 주변
FIGSIZE = (11.4, 4.5)
DPI = 100


def draw(i):
    wc, fc, kind = stages[i]
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0.058, 0.145, 0.632, 0.79])     # 전체
    axi = fig.add_axes([0.762, 0.145, 0.222, 0.79])    # Hβ 확대 (분리 배치)

    prev = stages[i - 1] if i > 0 else None

    # ── 전체 ──
    if prev is not None and prev[2] == kind:
        ax.plot(prev[0], prev[1], lw=0.7, color=PREV, zorder=1)
    ax.plot(wc, fc, lw=0.8, color=CUR, zorder=3)

    if kind == 'norm':
        ax.set_xlim(WAVE_MIN - 60, WAVE_MAX + 60)
        ax.set_ylim(0.35, 1.42)
        ax.axhline(1.0, color='#3a4049', lw=0.9, ls='--', zorder=2)
        ax.set_ylabel('정규화 flux')
        if i == 8:      # 무엇으로 나눴는지 — 이전 스펙트럼 + 연속선
            axt = ax.twinx()
            axt.plot(prev[0], prev[1], lw=0.6, color=PREV, zorder=0)
            axt.plot(prev[0], cont2, lw=1.6, color='#e05545', ls='--', zorder=1)
            axt.set_ylim(0, np.percentile(prev[1], 99.7) * 1.5)
            axt.set_yticks([])
            for s in axt.spines.values():
                s.set_visible(False)
            ax.set_zorder(axt.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.text(.015, .05, '회색 = 정규화 전   빨강 점선 = 연속선 추정',
                    transform=ax.transAxes, fontsize=9.5, color=DIM)
    else:
        ax.set_xlim(3660, 9140) if i < 5 else ax.set_xlim(WAVE_MIN - 60,
                                                          WAVE_MAX + 60)
        vis = fc[(wc >= ax.get_xlim()[0]) & (wc <= ax.get_xlim()[1])]
        if len(vis):
            hi = np.percentile(vis, 99.7)
            ax.set_ylim(-0.04 * hi, hi * 1.16)
        ax.set_ylabel('관측 flux (상대)')
        ax.set_yticks([])
    if i > 0 and prev is not None and prev[2] == kind:
        ax.text(.985, .04, '회색 = 이전 단계', transform=ax.transAxes,
                fontsize=9.5, color=DIM, ha='right')
    ax.set_xlabel('파장 (Å)')
    ax.grid(alpha=0.2, lw=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.55)

    # ── Hβ 확대 ──
    if prev is not None and prev[2] == kind:
        mp = (prev[0] >= ZOOM[0]) & (prev[0] <= ZOOM[1])
        if mp.sum() > 2:
            axi.plot(prev[0][mp], prev[1][mp], lw=1.1, color=PREV)
    mc = (wc >= ZOOM[0]) & (wc <= ZOOM[1])
    if mc.sum() > 2:
        axi.plot(wc[mc], fc[mc], lw=1.5, color=CUR)
    axi.axvline(4861.3, color='#3a4049', lw=0.8, ls=':')
    axi.set_xlim(*ZOOM)
    axi.set_xticks([4850, 4860, 4870])
    axi.tick_params(labelsize=9, length=2, colors=DIM)
    axi.set_yticks([])
    axi.grid(alpha=0.18, lw=0.5)
    for s in axi.spines.values():
        s.set_alpha(0.55)
    axi.set_xlabel('Hβ 4861 Å 확대', fontsize=10, color=DIM)

    fig.savefig(os.path.join(OUT, f'step{i}.png'), transparent=True, dpi=DPI)
    plt.close(fig)


for i in range(10):
    draw(i)

from PIL import Image
sizes = {Image.open(os.path.join(OUT, f'step{i}.png')).size for i in range(10)}
print('생성 10장, 크기 집합 =', sizes, '(1개면 정상)')
print(f'불량 화소 {n_bad}개 · RV {RV:.1f} km/s · A_V {AV} · SG 창 {win}')
