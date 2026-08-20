# -*- coding: utf-8 -*-
"""
build_av_v5_starhorse.py — A_V 결측 보완 (StarHorse 2021)

Gaia GSP-Phot(A0)은 37.5%만 커버 → 남은 별을 StarHorse2021(I/354,
Anders+2022, Gaia EDR3 기반 3.6억 별 베이지안 파라미터)의 AV50 으로 보완.
우선순위: azero_gspphot > StarHorse AV50 > NaN(미보정)

실행: python build_av_v5_starhorse.py  (build_av_v5.py 이후)
출력: master_catalog_v5.csv 갱신 (av/av_src 재계산)
"""

import os
import numpy as np
import pandas as pd

BASE = r'C:\Users\user\Desktop\최종\v5'
MAP_CACHE = os.path.join(BASE, 'cache_spec_gaia_map.csv')
AV_CACHE = os.path.join(BASE, 'cache_gaia_av.csv')
SH_CACHE = os.path.join(BASE, 'cache_starhorse_av.csv')
CATALOG = os.path.join(r'C:\Users\user\Desktop\최종',
                       'v5_인수인계', '04_데이터', 'master_catalog.csv')
OUT_CSV = os.path.join(BASE, 'master_catalog_v5.csv')
CHUNK = 50000
AG_TO_AV = 1.0 / 0.789


def main():
    mp = pd.read_csv(MAP_CACHE)
    av = pd.read_csv(AV_CACHE).drop_duplicates('source_id')
    av['source_id'] = av['source_id'].astype(np.int64)

    # A0 없는 source_id 만 StarHorse 조회
    have = set(av.loc[av['azero_gspphot'].notna(), 'source_id'])
    all_ids = set(mp.loc[mp['source_id'] > 0, 'source_id'].astype(np.int64))
    todo = np.array(sorted(all_ids - have), dtype=np.int64)

    done = set()
    if os.path.exists(SH_CACHE):
        done = set(pd.read_csv(SH_CACHE, usecols=['source_id'])
                   ['source_id'].astype(np.int64))
        print(f"StarHorse 캐시 {len(done):,}개 — 이어서")
    todo = np.array(sorted(set(todo) - done), dtype=np.int64)
    print(f"StarHorse 조회 대상: {len(todo):,}개")

    if len(todo):
        from astroquery.utils.tap.core import TapPlus
        from astropy.table import Table
        tap = TapPlus(url='https://tapvizier.cds.unistra.fr/TAPVizieR/tap')
        q = ('SELECT t.source_id, s."AV50" AS av50 '
             'FROM tap_upload.ids AS t '
             'LEFT JOIN "I/354/starhorse2021" AS s '
             'ON s."Source" = t.source_id')
        for s in range(0, len(todo), CHUNK):
            ids = todo[s:s + CHUNK]
            up = Table({'source_id': ids})
            print(f"  청크 {s//CHUNK + 1}/{(len(todo)-1)//CHUNK + 1} "
                  f"({len(ids):,}개)...", flush=True)
            for attempt in range(3):
                try:
                    job = tap.launch_job_async(
                        q, upload_resource=up, upload_table_name='ids')
                    r = job.get_results().to_pandas()
                    break
                except Exception as e:
                    print(f"    재시도 {attempt+1}/3: {type(e).__name__}")
                    if attempt == 2:
                        raise
            r.columns = [c.lower() for c in r.columns]
            r = r.drop_duplicates('source_id')
            r.to_csv(SH_CACHE, mode='a', index=False,
                     header=not os.path.exists(SH_CACHE))
            print(f"    수신 {len(r):,} (AV50 있음 "
                  f"{r['av50'].notna().sum():,})", flush=True)

    sh = pd.read_csv(SH_CACHE).drop_duplicates('source_id')
    sh['source_id'] = sh['source_id'].astype(np.int64)

    # 병합 + A_V 채택 재계산
    cat = pd.read_csv(CATALOG)
    cat = cat.merge(mp[['spec_id', 'source_id']], on='spec_id', how='left')
    cat['source_id'] = cat['source_id'].fillna(0).astype(np.int64)
    cat = cat.merge(av, on='source_id', how='left')
    cat = cat.merge(sh, on='source_id', how='left')

    azero = pd.to_numeric(cat['azero_gspphot'], errors='coerce')
    ag = pd.to_numeric(cat['ag_gspphot'], errors='coerce')
    av50 = pd.to_numeric(cat['av50'], errors='coerce')

    cat['av'] = azero
    cat['av'] = cat['av'].where(cat['av'].notna(), av50)
    cat['av'] = cat['av'].where(cat['av'].notna(), ag * AG_TO_AV)
    cat['av_src'] = np.where(azero.notna(), 'azero_gspphot',
                    np.where(av50.notna(), 'starhorse_av50',
                    np.where(ag.notna(), 'ag_gspphot', 'none')))

    cat.to_csv(OUT_CSV, index=False)
    n = cat['av'].notna().sum()
    print(f"\n저장: {OUT_CSV}")
    print(f"A_V 확보: {n:,}/{len(cat):,} ({100*n/len(cat):.1f}%)")
    print(cat['av_src'].value_counts().to_string())
    for src, grp in cat.groupby('source'):
        k = grp['av'].notna().sum()
        print(f"  {src}: {k:,}/{len(grp):,} ({100*k/len(grp):.1f}%)  "
              f"중앙값 {grp['av'].median():.3f}")


if __name__ == '__main__':
    main()
