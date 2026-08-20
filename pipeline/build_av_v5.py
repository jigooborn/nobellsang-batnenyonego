# -*- coding: utf-8 -*-
"""
build_av_v5.py — 전 서베이 A_V 확보 (v5 요구사항 ②: 성간소광 전체 적용)

절차:
  1) master_catalog.csv 의 spec_id 마다 Gaia DR3 source_id 를 붙인다
     - LAMOST : dr9_v2.0_LRS_stellar/mstellar.fits (obsid → gaia_source_id)
     - SEGUE  : segue1_catalog_gaia.csv (plate-mjd-fiber → source_id)
     - MILES  : miles_catalog_gaia.csv (MILES_ID → source_id)
     - MaStar : mastarall-gaiaedr3 fits GOODSTARS (MANGAID → SOURCE_ID)
  2) 고유 source_id 를 Gaia 아카이브 TAP 에 청크 업로드 조인으로
     azero_gspphot(A_0)·ag_gspphot(A_G) 조회 (익명, 5만 개씩)
  3) A_V 채택: azero_gspphot 우선 (541.4nm 단색 소광 — A_V 에 근사),
     없으면 A_G/0.789 (Wang & Chen 2019 환산), 그것도 없으면 NaN(미보정)
  4) master_catalog_v5.csv 저장 (기존 컬럼 + source_id + av 컬럼)

중간 산출물은 전부 캐시되어 재실행 시 이어서 진행:
  cache_spec_gaia_map.csv  (1 결과)
  cache_gaia_av.csv        (2 결과, 청크 단위 append)
"""

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits

BASE     = r'C:\Users\user\Desktop\최종'
RAW      = os.path.join(BASE, '스펙트럼원본')
CATALOG  = os.path.join(BASE, 'v5_인수인계', '04_데이터', 'master_catalog.csv')
OUT_DIR  = os.path.join(BASE, 'v5')
MAP_CACHE = os.path.join(OUT_DIR, 'cache_spec_gaia_map.csv')
AV_CACHE  = os.path.join(OUT_DIR, 'cache_gaia_av.csv')
OUT_CSV   = os.path.join(OUT_DIR, 'master_catalog_v5.csv')

CHUNK = 50000          # TAP 업로드 조인 청크 크기
AG_TO_AV = 1.0 / 0.789 # A_G → A_V (Wang & Chen 2019)


def build_gaia_map():
    if os.path.exists(MAP_CACHE):
        m = pd.read_csv(MAP_CACHE)
        print(f"[1] 캐시 사용: {MAP_CACHE} ({len(m):,}행)")
        return m
    cat = pd.read_csv(CATALOG, usecols=['spec_id'])
    sid = cat['spec_id']

    frames = []

    # LAMOST — stellar + mstellar
    print("[1] LAMOST obsid → gaia_source_id ...")
    lam_map = {}
    for f in ['dr9_v2.0_LRS_stellar.fits', 'dr9_v2.0_LRS_mstellar.fits']:
        with fits.open(os.path.join(RAW, 'lamost(중국)', f), memmap=True) as h:
            d = h[1].data
            obsid = np.asarray(d['obsid'], dtype=np.int64)
            gid = np.asarray(d['gaia_source_id'], dtype=np.int64)
        for o, g in zip(obsid, gid):
            if g > 0:
                lam_map[int(o)] = int(g)
        print(f"    {f}: 누적 {len(lam_map):,}")
    lam = sid[sid.str.startswith('LAMOST-')]
    obs = lam.str.replace('LAMOST-', '', regex=False).astype(np.int64)
    frames.append(pd.DataFrame({
        'spec_id': lam.values,
        'source_id': [lam_map.get(int(o), 0) for o in obs]}))

    # SEGUE
    print("[1] SEGUE ...")
    seg_cat = pd.read_csv(os.path.join(
        RAW, 'segue1_spectra(sdss)', 'segue1_catalog_gaia.csv'))
    seg_cat['spec_id'] = ('SEGUE-' + seg_cat['plate'].astype(int).astype(str)
                          + '-' + seg_cat['mjd'].astype(int).astype(str)
                          + '-' + seg_cat['fiberid'].astype(int).astype(str))
    seg_cat['source_id'] = (pd.to_numeric(seg_cat['source_id'],
                                          errors='coerce')
                            .fillna(0).astype(np.int64))
    frames.append(seg_cat[['spec_id', 'source_id']]
                  .drop_duplicates('spec_id'))

    # MILES
    print("[1] MILES ...")
    mil = pd.read_csv(os.path.join(RAW, 'miles', 'miles_catalog_gaia.csv'))
    mil['spec_id'] = 'MILES-' + mil['MILES_ID'].astype(int).map('{:04d}'.format)
    mil['source_id'] = (pd.to_numeric(mil['source_id'], errors='coerce')
                        .fillna(0).astype(np.int64))
    frames.append(mil[['spec_id', 'source_id']].drop_duplicates('spec_id'))

    # MaStar — GOODSTARS (별 단위; spec_id 는 MASTAR-<MANGAID>)
    print("[1] MaStar ...")
    with fits.open(os.path.join(
            RAW, 'mastar(sdss)',
            'mastarall-gaiaedr3-extcorr-simbad-ps1-v3_1_1-v1_7_7-v1.fits'),
            memmap=True) as h:
        d = h['GOODSTARS'].data
        mid = [str(x).strip() for x in d['MANGAID']]
        gid = np.asarray(d['SOURCE_ID'], dtype=np.int64)
    frames.append(pd.DataFrame({
        'spec_id': ['MASTAR-' + m for m in mid],
        'source_id': gid}).drop_duplicates('spec_id'))

    mp = pd.concat(frames, ignore_index=True)
    mp = cat.merge(mp, on='spec_id', how='left')
    mp['source_id'] = mp['source_id'].fillna(0).astype(np.int64)
    mp.to_csv(MAP_CACHE, index=False)
    n = (mp['source_id'] > 0).sum()
    print(f"[1] 완료 — source_id 확보 {n:,}/{len(mp):,} "
          f"({100*n/len(mp):.1f}%) → {MAP_CACHE}")
    return mp


def query_gaia_av(source_ids):
    """고유 source_id 목록 → A0/AG 조회 (청크, 캐시 append).

    ESA Gaia 아카이브가 이 네트워크에서 SSL 차단되어 있어서
    VizieR TAP 미러(I/355/paramp = gaiadr3.astrophysical_parameters,
    A0=azero_gspphot, AG=ag_gspphot)를 사용한다."""
    from astroquery.utils.tap.core import TapPlus
    from astropy.table import Table

    done = set()
    if os.path.exists(AV_CACHE):
        done = set(pd.read_csv(AV_CACHE, usecols=['source_id'])
                   ['source_id'].astype(np.int64))
        print(f"[2] 캐시에 {len(done):,}개 있음 — 이어서 진행")

    todo = np.array(sorted(set(source_ids) - done), dtype=np.int64)
    todo = todo[todo > 0]
    print(f"[2] 조회할 source_id: {len(todo):,}개 (청크 {CHUNK:,})")

    tap = TapPlus(url='https://tapvizier.cds.unistra.fr/TAPVizieR/tap')
    q = ('SELECT t.source_id, p."A0" AS azero_gspphot, '
         'p."AG" AS ag_gspphot '
         'FROM tap_upload.ids AS t '
         'LEFT JOIN "I/355/paramp" AS p ON p."Source" = t.source_id')
    for s in range(0, len(todo), CHUNK):
        ids = todo[s:s + CHUNK]
        up = Table({'source_id': ids})
        print(f"  청크 {s//CHUNK + 1}/{(len(todo)-1)//CHUNK + 1} "
              f"({len(ids):,}개) 조회 중...", flush=True)
        for attempt in range(3):
            try:
                job = tap.launch_job_async(
                    q, upload_resource=up, upload_table_name='ids')
                r = job.get_results().to_pandas()
                break
            except Exception as e:
                print(f"    재시도 {attempt+1}/3: {type(e).__name__}",
                      flush=True)
                if attempt == 2:
                    raise
        r.columns = [c.lower() for c in r.columns]
        r = r.drop_duplicates('source_id')
        r.to_csv(AV_CACHE, mode='a', index=False,
                 header=not os.path.exists(AV_CACHE))
        print(f"    수신 {len(r):,}행 (A0 있음 "
              f"{r['azero_gspphot'].notna().sum():,})", flush=True)
    return pd.read_csv(AV_CACHE).drop_duplicates('source_id')


def main():
    mp = build_gaia_map()
    av = query_gaia_av(mp.loc[mp['source_id'] > 0, 'source_id'].values)
    av['source_id'] = av['source_id'].astype(np.int64)

    cat = pd.read_csv(CATALOG)
    cat = cat.merge(mp, on='spec_id', how='left')
    cat = cat.merge(av, on='source_id', how='left')

    # A_V 채택: azero 우선 → ag/0.789 → NaN
    azero = pd.to_numeric(cat['azero_gspphot'], errors='coerce')
    ag = pd.to_numeric(cat['ag_gspphot'], errors='coerce')
    cat['av'] = azero.where(azero.notna(), ag * AG_TO_AV)
    cat['av_src'] = np.where(azero.notna(), 'azero_gspphot',
                    np.where(ag.notna(), 'ag_gspphot', 'none'))

    cat.to_csv(OUT_CSV, index=False)
    n = cat['av'].notna().sum()
    print(f"\n[3] 저장: {OUT_CSV}")
    print(f"    A_V 확보: {n:,}/{len(cat):,} ({100*n/len(cat):.1f}%)")
    print(f"    소스별 커버리지:")
    for src, grp in cat.groupby('source'):
        k = grp['av'].notna().sum()
        print(f"      {src}: {k:,}/{len(grp):,} ({100*k/len(grp):.1f}%)  "
              f"A_V 중앙값 {grp['av'].median():.3f}")


if __name__ == '__main__':
    main()
