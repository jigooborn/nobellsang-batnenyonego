# -*- coding: utf-8 -*-
"""
select_m_targets.py — M형 보강 대상 15,000개 선정 (LAMOST DR9 mstellar)

기준:
  - LASP-M 파이프라인 Teff ≤ 3,425K (경계 3,500K − 버퍼 75K → 확실한 M)
  - SNR(r 또는 i) > 10  (기존 다운로드 기준과 동일)
  - 이미 보유한 obsid / 카탈로그 등재 obsid 제외
  - 거성(gM)·왜성(dM) 모두 포함 — XSL 실패가 거성 쪽이므로 거성 우선 확보
  - 선별: 서브클래스(dM/gM) 층화 + SNR 내림차순 (함정 2번 준수)
출력: m_targets.csv (obsid, teff, logg, m_h, snr, subclass,
                     gaia_source_id, ra, dec)
"""
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits

BASE = r'C:\Users\user\Desktop\최종'
MSTELLAR = os.path.join(BASE, '스펙트럼원본', 'lamost(중국)',
                        'dr9_v2.0_LRS_mstellar.fits')
SPECDIR = os.path.join(BASE, '스펙트럼원본', 'lamost(중국)',
                       'lamost_spectra')
CATALOG = os.path.join(BASE, 'v5', 'master_catalog_v5.csv')
OUT = os.path.join(BASE, 'v5', 'm_targets.csv')

N_TARGET = 15000
TEFF_MAX = 3425.0
SNR_MIN = 10.0


def main():
    with fits.open(MSTELLAR, memmap=True) as h:
        d = h[1].data
        df = pd.DataFrame({
            'obsid': np.asarray(d['obsid'], dtype=np.int64),
            'teff': np.asarray(d['teff'], dtype=np.float64),
            'logg': np.asarray(d['logg'], dtype=np.float64),
            'm_h': np.asarray(d['m_h'], dtype=np.float64),
            'snrr': np.asarray(d['snrr'], dtype=np.float64),
            'snri': np.asarray(d['snri'], dtype=np.float64),
            'subclass': [str(x).strip() for x in d['subclass']],
            'gaia_source_id': np.asarray(d['gaia_source_id'],
                                         dtype=np.int64),
            'ra': np.asarray(d['ra'], dtype=np.float64),
            'dec': np.asarray(d['dec'], dtype=np.float64),
        })
    print(f"mstellar 전체: {len(df):,}")

    df['snr'] = df[['snrr', 'snri']].max(axis=1)
    ok = (np.isfinite(df['teff']) & (df['teff'] > 2000)
          & (df['teff'] <= TEFF_MAX) & (df['snr'] > SNR_MIN))
    df = df[ok]
    print(f"Teff≤{TEFF_MAX:.0f}K & SNR>{SNR_MIN:.0f}: {len(df):,}")

    have = set()
    for f in glob.glob(os.path.join(SPECDIR, 'spec_*.fits')):
        try:
            have.add(int(os.path.basename(f)[5:-5]))
        except ValueError:
            pass
    cat = pd.read_csv(CATALOG, usecols=['spec_id'])
    cat_obs = set(int(s[7:]) for s in cat['spec_id']
                  if s.startswith('LAMOST-') and s[7:].isdigit())
    df = df[~df['obsid'].isin(have | cat_obs)]
    print(f"기존 보유/등재 제외 후: {len(df):,}")

    # 같은 별(gaia id) 중복 관측은 1개만 (SNR 최고)
    df = df.sort_values('snr', ascending=False)
    with_id = df[df['gaia_source_id'] > 0].drop_duplicates(
        'gaia_source_id')
    no_id = df[df['gaia_source_id'] <= 0]
    df = pd.concat([with_id, no_id])
    print(f"별 단위 중복 제거: {len(df):,}")

    # 거성/왜성 층화: 거성(gM)은 전부(희소), 나머지는 왜성 SNR 순
    is_giant = df['subclass'].str.startswith('g')
    giants = df[is_giant]
    dwarfs = df[~is_giant].sort_values('snr', ascending=False)
    n_g = min(len(giants), N_TARGET // 2)
    sel = pd.concat([giants.sort_values('snr', ascending=False).head(n_g),
                     dwarfs.head(N_TARGET - n_g)])
    print(f"선정: {len(sel):,} (거성 {is_giant[sel.index].sum():,} / "
          f"왜성 {(~is_giant[sel.index]).sum():,})")
    print(f"Teff 분포: {sel['teff'].min():.0f}~{sel['teff'].max():.0f}K, "
          f"중앙 {sel['teff'].median():.0f}K")
    sel.to_csv(OUT, index=False)
    print(f"저장: {OUT}")


if __name__ == '__main__':
    main()
