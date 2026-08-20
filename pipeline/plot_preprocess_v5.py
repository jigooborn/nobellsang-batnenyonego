# -*- coding: utf-8 -*-
"""
plot_preprocess_v5.py — 실제 스펙트럼으로 전처리 단계별 전/후 그림

preprocess_core 의 함수·상수를 그대로 사용해 process_one 의 순서를
단계별로 나눠 시각화 (보고서 '데이터 전처리' 섹션 그림).
대상: LAMOST K형 별 (흡수선 풍부 + A_V 큰 별 → 소광 보정 효과가 보임)

실행: python plot_preprocess_v5.py [spec_id]
출력: results/figures_v51/5_preprocess_steps.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from preprocess_core import (C_KMS, WAVE_MIN, WAVE_MAX, N_PIX, TARGET_WAVE,
                             SG_WIN, CONT_ITERS, SIGMA_CUT, KSIG_ITERS,
                             CONT_FLOOR, fit_continuum, kappa_sigma_clip,
                             cardelli_alav, vac_to_air, safe_arr)
from classify_gui_v5 import SpectrumSource

BASE = r'C:\Users\user\Desktop\최종'
SPECDIR = os.path.join(BASE, '스펙트럼원본', 'lamost(중국)',
                       'lamost_spectra')
OUT = os.path.join('results', 'figures_v51')

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')
except Exception:
    pass


def pick_star():
    cat = pd.read_csv('master_catalog_v5.csv',
                      usecols=['spec_id', 'source', 'teff_adopted',
                               'snr', 'av'])
    av = pd.to_numeric(cat['av'], errors='coerce')
    snr = pd.to_numeric(cat['snr'], errors='coerce')
    sel = cat[(cat['source'] == 'LAMOST') & (av > 0.8) & (av < 2.0)
              & (snr > 40)
              & (cat['teff_adopted'] > 4000)
              & (cat['teff_adopted'] < 5200)].copy()
    sel['av'] = av[sel.index]
    sel = sel.sort_values('snr', ascending=False)
    # 시연 적합성 검사: 빨간 끝이 살아있고(死픽셀 구간 없음),
    # 도플러 이동이 눈에 보일 만큼 RV 가 큰 별 (|RV|≥60 km/s ≈ 1.3Å@Hα)
    fallback = None
    for _, r in sel.head(400).iterrows():
        obsid = r['spec_id'].replace('LAMOST-', '')
        p = os.path.join(SPECDIR, f'spec_{obsid}.fits')
        if not os.path.exists(p):
            continue
        try:
            s = SpectrumSource(p)
            _, w, f, rv, _ = s.get(0)
            s.close()
        except Exception:
            continue
        if rv is None:
            continue
        mred = (w > 7000) & (w < 7400)
        mall = (w > 4000) & (w < 7400)
        if not mred.any():
            continue
        red_ok = (np.nanmedian(f[mred])
                  > 0.3 * np.nanmedian(f[mall]))
        if not red_ok:
            continue
        cand = (p, r['spec_id'], float(r['av']),
                float(r['teff_adopted']))
        if fallback is None:
            fallback = cand
        if abs(rv) >= 60:
            return cand
    if fallback:
        return fallback
    raise RuntimeError('조건 맞는 별 없음')


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        sid = sys.argv[1]
        path = os.path.join(SPECDIR,
                            f"spec_{sid.replace('LAMOST-', '')}.fits")
        cat = pd.read_csv('master_catalog_v5.csv',
                          usecols=['spec_id', 'teff_adopted', 'av'])
        row = cat[cat['spec_id'] == sid].iloc[0]
        av, teff = float(row['av']), float(row['teff_adopted'])
    else:
        path, sid, av, teff = pick_star()
    print(f"대상: {sid}  Teff={teff:.0f}K  A_V={av:.2f}")

    src = SpectrumSource(path)
    _, wave0, flux0, rv, _ = src.get(0)
    print(f"RV(헤더) = {rv:.1f} km/s")

    # 리더가 이미 진공→공기 변환을 하므로, 0단계 시각화용 진공 파장은
    # fits 에서 직접 읽음 (LAMOST 원본은 진공 파장)
    from astropy.io import fits as _fits
    wave_vac = None
    with _fits.open(path, memmap=True) as h:
        if len(h) > 1 and hasattr(h[1], 'columns') and h[1].columns:
            names = h[1].columns.names
            low = [c.lower() for c in names]
            if 'wavelength' in low:
                wave_vac = safe_arr(
                    h[1].data[names[low.index('wavelength')]]).flatten()
            elif 'loglam' in low:
                wave_vac = 10.0 ** safe_arr(
                    h[1].data[names[low.index('loglam')]]).flatten()
        if wave_vac is None and h[0].data is not None:
            hd = h[0].header
            arr = safe_arr(h[0].data)
            n = arr.shape[-1]
            wave_vac = 10.0 ** (hd['COEFF0'] + hd['COEFF1'] * np.arange(n))

    # ── process_one 순서를 단계별로 재현 (같은 함수·상수) ──
    wave, flux = wave0.copy(), flux0.copy()
    fw = np.isfinite(wave)
    wave, flux = wave[fw], flux[fw]

    # 1) 배드픽셀 보간
    good = np.isfinite(flux) & (flux > 0)
    flux_bp = flux.copy()
    gi = np.where(good)[0]
    flux_bp = np.interp(np.arange(len(flux)), gi, flux[gi])
    n_bad = int((~good).sum())

    # 2) 도플러 보정
    wave_dop = wave / (1.0 + rv / C_KMS)

    # 3) 소광 보정
    flux_ext = flux_bp * 10.0 ** (0.4 * av * cardelli_alav(wave_dop))

    # 4) 파장 컷
    m = (wave_dop >= WAVE_MIN) & (wave_dop <= WAVE_MAX)
    wc, fc_before_ks = wave_dop[m], flux_ext[m]
    order = np.argsort(wc)
    wc, fc_before_ks = wc[order], fc_before_ks[order]
    wn = (wc - WAVE_MIN) / (WAVE_MAX - WAVE_MIN) * 2.0 - 1.0

    # 5) κ-σ 클리핑
    cont1 = fit_continuum(wn, fc_before_ks, deg=3, iters=CONT_ITERS)
    fc_ks = kappa_sigma_clip(fc_before_ks, cont1, kappa=SIGMA_CUT,
                             iters=KSIG_ITERS)

    # 6) SG 스무딩
    fc_sg = savgol_filter(fc_ks, window_length=SG_WIN, polyorder=3)

    # 7) 연속선 피팅·정규화
    cont2 = fit_continuum(wn, fc_sg, deg=3, iters=CONT_ITERS)
    med = np.median(cont2)
    lowc = cont2 < CONT_FLOOR * med
    fnorm = fc_sg / np.where(lowc, med, cont2)
    fnorm[lowc] = 1.0

    # 8) 리샘플
    fout = np.interp(TARGET_WAVE, wc, fnorm, left=1.0, right=1.0)
    fout = np.clip(fout, 0.0, 5.0)

    # ── 그림: 3행 3열 (9패널) ──
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    axes = axes.flatten()

    ax = axes[0]
    if wave_vac is not None and len(wave_vac) == len(flux0):
        mv0 = np.isfinite(flux0) & np.isfinite(wave_vac)
        ax.plot(wave_vac[mv0], flux0[mv0], color='#777', lw=0.5)
    else:
        mm0 = np.isfinite(flux)
        ax.plot(wave[mm0], flux[mm0], color='#777', lw=0.5)
    ax.set_title(f"① 원본 관측 (진공 파장 기준)\n{sid} · "
                 f"배드픽셀 {n_bad}개 포함", fontsize=11,
                 fontweight='bold')
    ax.set_ylabel('flux')

    ax = axes[1]
    zv = (6550, 6580)
    if wave_vac is not None:
        mv = (wave_vac > zv[0] - 15) & (wave_vac < zv[1] + 15)
        ax.plot(wave_vac[mv], flux0[mv], color='#999', lw=1.2,
                label='진공 파장 (SDSS·LAMOST 원본)')
        wa = vac_to_air(wave_vac)
        ax.plot(wa[mv], flux0[mv], color='#1f77b4', lw=1.2,
                label='공기 파장 (변환 후)')
        dl = float(np.mean(wave_vac[mv] - wa[mv]))
        ax.set_xlim(*zv)
        ax.set_title(f"② 진공→공기 파장 변환 (Morton 1991)\n"
                     f"모든 선이 ~{dl:.1f}Å 이동 — MILES(공기)와 기준 통일",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(wave, flux_bp, color='#1f77b4', lw=0.5,
            label='보간 후')
    if n_bad:
        bad_w = wave[~good]
        ax.scatter(bad_w, np.interp(bad_w, wave, flux_bp), s=12,
                   color='#c0392b', zorder=5,
                   label=f'보간된 배드픽셀 ({n_bad}개)')
    ax.set_title("③ 배드픽셀(NaN·0·음수) 이웃 보간\n— 가짜 0-딥 흡수선 방지",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[3]
    zw = (6540, 6590)
    mz = (wave > zw[0] - 20) & (wave < zw[1] + 20)
    ax.plot(wave[mz], flux_bp[mz], color='#999', lw=1.2,
            label='보정 전 (관측 파장)')
    mz2 = (wave_dop > zw[0] - 20) & (wave_dop < zw[1] + 20)
    ax.plot(wave_dop[mz2], flux_bp[mz2], color='#1f77b4', lw=1.2,
            label='보정 후 (정지 좌표)')
    ax.axvline(6562.8, color='#c0392b', ls='--', lw=1,
               label='Hα 정지 파장 6562.8Å')
    ax.set_xlim(*zw)
    ax.set_title(f"④ 도플러(시선속도) 보정 — RV {rv:.0f} km/s\n"
                 "Hα 확대: 흡수선이 정지 파장으로 이동", fontsize=11,
                 fontweight='bold')
    ax.set_ylabel('flux')
    ax.legend(fontsize=9)

    ax = axes[4]
    ax.plot(wc, fc_before_ks / np.median(fc_before_ks), color='#1f77b4',
            lw=0.6, label=f'소광 보정 후 (A_V={av:.2f})')
    mz3 = (wave_dop >= WAVE_MIN) & (wave_dop <= WAVE_MAX)
    ax.plot(wave_dop[mz3], flux_bp[mz3] / np.median(flux_bp[mz3]),
            color='#999', lw=0.6, label='보정 전 (붉게 왜곡됨)')
    ax.set_title("⑤ 성간소광 보정 (Cardelli, R_V=3.1)\n"
                 "— 먼지가 흡수한 파란빛을 복원 → 연속선 기울기 회복",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[5]
    resid_spikes = fc_before_ks > cont1 * (1 + SIGMA_CUT *
        np.std(fc_before_ks / cont1 - 1))
    ax.plot(wc, fc_before_ks, color='#999', lw=0.5, label='클리핑 전')
    ax.plot(wc, fc_ks, color='#1f77b4', lw=0.5, alpha=0.8,
            label='클리핑 후')
    if resid_spikes.any():
        ax.scatter(wc[resid_spikes], fc_before_ks[resid_spikes], s=14,
                   color='#c0392b', zorder=5,
                   label=f'제거된 스파이크 ({int(resid_spikes.sum())}px)')
    ax.set_title("⑥ κ-σ 클리핑 (+3σ 위쪽만, 3회)\n"
                 "— 우주선 스파이크 제거, 아래로 파인 흡수선은 보존",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('flux')
    ax.legend(fontsize=9)

    ax = axes[6]
    zb = (4205, 4250)
    mzb = (wc > zb[0]) & (wc < zb[1])
    norm_loc = np.median(fc_ks[mzb])
    ax.plot(wc[mzb], fc_ks[mzb] / norm_loc, color='#999',
            lw=1.0, label='스무딩 전 (노이즈 포함)')
    ax.plot(wc[mzb], fc_sg[mzb] / norm_loc, color='#1f77b4',
            lw=1.6, label=f'SG 스무딩 후 (win={SG_WIN})')
    ax.axvline(4226.7, color='#c0392b', ls='--', lw=1,
               label='Ca I 4227 (K형 지표선)')
    ax.set_title("⑦ Savitzky-Golay 스무딩 (윈도우 9px, 3차)\n"
                 "Ca I 4227 확대 — 노이즈는 줄이되 선의 깊이·형태는 보존",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[7]
    ax.plot(wc, fc_sg, color='#1f77b4', lw=0.5, label='스펙트럼')
    ax.plot(wc, cont2, color='#c0392b', lw=2,
            label='추정된 연속선 (3차 다항 반복 피팅)')
    ax.set_title("⑧ 연속선 추정 — 이 곡선으로 나눠서 정규화\n"
                 "(개형 정보는 색지수 2개로 별도 보존)", fontsize=11,
                 fontweight='bold')
    ax.set_xlabel('파장 (Å)')
    ax.set_ylabel('flux')
    ax.legend(fontsize=9)

    ax = axes[8]
    ax.plot(TARGET_WAVE, fout, color='#1f77b4', lw=0.6)
    ax.axhline(1.0, color='#999', ls='--', lw=0.7)
    ax.set_title(f"⑨ 정규화 + 공통 격자 리샘플 ({N_PIX}px, 1Å/px)\n"
                 "— 모델 입력 완성: 흡수선의 상대 깊이만 남음",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('파장 (Å)')
    ax.set_ylabel('정규화 flux')

    fig.suptitle(f"전처리 파이프라인 단계별 실측 — {sid} "
                 f"(K형, Teff {teff:.0f}K, A_V {av:.2f}, RV {rv:.0f}km/s)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT, '5_preprocess_steps.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"저장: {out}")


if __name__ == '__main__':
    main()
