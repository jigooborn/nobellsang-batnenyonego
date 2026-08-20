# -*- coding: utf-8 -*-
"""
append_catalog_m.py — 보강 M형을 master_catalog_v5 에 병합

- 라벨: mstellar LASP-M 파이프라인의 teff/logg/m_h (SIMBAD 아님 — 스펙트럼 피팅)
- 별 그룹: 기존 카탈로그에 같은 gaia_source_id 가 있으면 그 group_id 재사용
  (같은 별이 train/test 양쪽에 못 가게 — 누수 방지), 없으면 새 그룹
- A_V: 기존 캐시 재사용 + 새 source_id 만 VizieR(A0→StarHorse) 추가 조회
실행: python append_catalog_m.py  (다운로드 완료 후)
"""
import os
import socket
import numpy as np
import pandas as pd

# astroquery TAP 은 자체 타임아웃이 없어 죽은 연결에 무한 대기 가능
# (실측: 26분 행) → 전역 소켓 타임아웃으로 상한 강제
socket.setdefaulttimeout(60)

BASE = r'C:\Users\user\Desktop\최종'
V5 = os.path.join(BASE, 'v5')
TARGETS = os.path.join(V5, 'm_targets.csv')
SPECDIR = os.path.join(BASE, '스펙트럼원본', 'lamost(중국)',
                       'lamost_spectra')
CATALOG = os.path.join(V5, 'master_catalog_v5.csv')
AV_CACHE = os.path.join(V5, 'cache_gaia_av.csv')
SH_CACHE = os.path.join(V5, 'cache_starhorse_av.csv')
AG_TO_AV = 1.0 / 0.789
CHUNK = 50000


def query_vizier(ids, table, id_col, val_cols, cache_path):
    """VizieR 업로드 조인 (build_av_v5 와 동일 방식, 캐시 append)."""
    done = set()
    if os.path.exists(cache_path):
        done = set(pd.read_csv(cache_path, usecols=['source_id'])
                   ['source_id'].astype(np.int64))
    todo = np.array(sorted(set(ids) - done), dtype=np.int64)
    todo = todo[todo > 0]
    if not len(todo):
        return
    from astroquery.utils.tap.core import TapPlus
    from astropy.table import Table
    tap = TapPlus(url='https://tapvizier.cds.unistra.fr/TAPVizieR/tap')
    sel = ", ".join(f'p."{c}" AS {a}' for c, a in val_cols)
    q = (f'SELECT t.source_id, {sel} FROM tap_upload.ids AS t '
         f'LEFT JOIN "{table}" AS p ON p."{id_col}" = t.source_id')
    import time
    for s in range(0, len(todo), CHUNK):
        up = Table({'source_id': todo[s:s + CHUNK]})
        print(f"  [{table}] 청크 {s//CHUNK+1} ({len(up):,}개)...",
              flush=True)
        r = None
        for attempt in range(6):     # DNS 간헐 단절 대비: 점증 대기 재시도
            try:
                job = tap.launch_job_async(q, upload_resource=up,
                                           upload_table_name='ids')
                r = job.get_results().to_pandas()
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"    재시도 {attempt+1}/6 ({type(e).__name__}) — "
                      f"{wait}초 대기", flush=True)
                time.sleep(wait)
        if r is None:
            print(f"    !! [{table}] 조회 포기 — 해당 별들은 미보정 처리")
            continue
        r.columns = [c.lower() for c in r.columns]
        r = r.drop_duplicates('source_id')
        r.to_csv(cache_path, mode='a', index=False,
                 header=not os.path.exists(cache_path))


def main():
    cat = pd.read_csv(CATALOG)
    tg = pd.read_csv(TARGETS)

    # 실제로 받아진 파일만
    have = []
    for _, r in tg.iterrows():
        p = os.path.join(SPECDIR, f"spec_{int(r['obsid'])}.fits")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            have.append(r)
    tg = pd.DataFrame(have)
    print(f"다운로드 확인: {len(tg):,}개")

    tg['spec_id'] = 'LAMOST-' + tg['obsid'].astype(int).astype(str)
    tg = tg[~tg['spec_id'].isin(set(cat['spec_id']))]
    print(f"카탈로그 신규: {len(tg):,}개")

    # A_V 조회 (신규 source_id 만). --offline 이면 온라인 조회 생략하고
    # 기존 캐시에 있는 값만 사용 (없는 별은 미보정 — M왜성은 근거리라
    # 소광이 작아 영향 미미, 네트워크 불안정 시의 안전 경로)
    import sys as _sys
    offline = '--offline' in _sys.argv
    sids = tg.loc[tg['gaia_source_id'] > 0,
                  'gaia_source_id'].astype(np.int64)
    if offline:
        print("!! --offline: A_V 온라인 조회 생략 (캐시만 사용)")
    else:
        query_vizier(sids, 'I/355/paramp', 'Source',
                     [('A0', 'azero_gspphot'), ('AG', 'ag_gspphot')],
                     AV_CACHE)
    av = pd.read_csv(AV_CACHE).drop_duplicates('source_id')
    av['source_id'] = av['source_id'].astype(np.int64)
    if not offline:
        need_sh = sids[~sids.isin(
            set(av.loc[pd.to_numeric(av['azero_gspphot'],
                                     errors='coerce').notna(),
                       'source_id']))]
        query_vizier(need_sh, 'I/354/starhorse2021', 'Source',
                     [('AV50', 'av50')], SH_CACHE)
    sh = pd.read_csv(SH_CACHE).drop_duplicates('source_id')
    sh['source_id'] = sh['source_id'].astype(np.int64)

    m = tg.merge(av, left_on='gaia_source_id', right_on='source_id',
                 how='left')
    m = m.merge(sh[['source_id', 'av50']], left_on='gaia_source_id',
                right_on='source_id', how='left', suffixes=('', '_sh'))
    azero = pd.to_numeric(m['azero_gspphot'], errors='coerce')
    ag = pd.to_numeric(m['ag_gspphot'], errors='coerce')
    av50 = pd.to_numeric(m['av50'], errors='coerce')
    m['av'] = azero.where(azero.notna(),
                          av50.where(av50.notna(), ag * AG_TO_AV))
    m['av_src'] = np.where(azero.notna(), 'azero_gspphot',
                  np.where(av50.notna(), 'starhorse_av50',
                  np.where(ag.notna(), 'ag_gspphot', 'none')))
    print(f"A_V 확보: {m['av'].notna().sum():,}/{len(m):,}")

    # 별 그룹: 기존 카탈로그의 같은 gaia source → 그룹 재사용
    sid2grp = {}
    if 'source_id' in cat.columns:
        c = cat[pd.to_numeric(cat['source_id'],
                              errors='coerce').fillna(0) > 0]
        sid2grp = dict(zip(c['source_id'].astype(np.int64),
                           c['group_id']))
    def grp(row):
        s = int(row['gaia_source_id'])
        if s > 0 and s in sid2grp:
            return sid2grp[s]
        return (f"mnew_g{s}" if s > 0
                else f"mnew_o{int(row['obsid'])}")
    m['group_id'] = m.apply(grp, axis=1)
    n_reuse = m['group_id'].isin(set(cat['group_id'])).sum()
    print(f"기존 그룹 재사용(중복 별): {n_reuse:,}개")

    new = pd.DataFrame({
        'spec_id': m['spec_id'], 'source': 'LAMOST',
        'group_id': m['group_id'],
        'ra': m['ra'], 'dec': m['dec'],
        'teff_adopted': m['teff'], 'logg_adopted': m['logg'],
        'feh_adopted': m['m_h'],
        'class': 'M', 'adopted_src': 'LAMOST_MSTELLAR',
        'teff': m['teff'], 'label_src': 'LAMOST_MSTELLAR',
        'teff_err': np.nan, 'snr': m['snr'], 'file_ref': '',
        'source_id': m['gaia_source_id'],
        'azero_gspphot': azero, 'ag_gspphot': ag, 'av50': av50,
        'av': m['av'], 'av_src': m['av_src'],
    })
    for col in cat.columns:
        if col not in new.columns:
            new[col] = np.nan
    out = pd.concat([cat, new[cat.columns]], ignore_index=True)
    out.to_csv(CATALOG, index=False)
    print(f"병합 완료: {len(cat):,} + {len(new):,} = {len(out):,}행 "
          f"→ {CATALOG}")


if __name__ == '__main__':
    main()
