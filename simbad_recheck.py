# -*- coding: utf-8 -*-
"""
simbad_recheck.py — SIMBAD 분광형 라벨 재검증 (자문 핵심 제안의 구현)

연구 프레이밍: "여러 시대·관측 프로그램에서 이질적으로 축적된 SIMBAD
분광형 라벨을, 균일한 최신 스펙트럼(MILES)으로 학습한 AI 로 독립
재분류하여 일치/불일치를 정량화하고, 불일치의 원인을 물리적으로 분석"

절차:
  1) MILES 985개 별의 SIMBAD 식별자 → SIMBAD TAP 에서 현재 등록된
     분광형(sp_type) 조회 (캐시됨)
  2) AI 예측(results/miles_batch_v5.csv)과 문자 비교
     (O·B 는 OB 로 묶어 6클래스 기준)
  3) 불일치 사례를 세 갈래로 분해:
     a. AI ↔ SIMBAD 불일치인데 카탈로그 Teff 는 AI 편  → SIMBAD 재검토 후보
     b. AI ↔ SIMBAD 불일치이고 Teff 도 SIMBAD 편       → AI 오분류
     c. 경계 인접(±1클래스) 불일치                      → 정의 차이 수준
출력: results/simbad_recheck/ (요약 txt + 재검토 후보 csv)
"""

import os
import re
import numpy as np
import pandas as pd

from preprocess_core import teff_to_class, CLASS_ORDER

BASE = r'C:\Users\user\Desktop\최종'
MILES_GAIA = os.path.join(BASE, '스펙트럼원본', 'miles',
                          'miles_catalog_gaia.csv')
PRED_CSV = os.path.join(BASE, 'v5', 'results', 'miles_batch_v5.csv')
SIMBAD_CACHE = os.path.join(BASE, 'v5', 'cache_simbad_sptype.csv')
OUT_DIR = os.path.join(BASE, 'v5', 'results', 'simbad_recheck')


def fetch_simbad_sptypes(names):
    """SIMBAD TAP sync 를 직접 호출(csv)해 식별자별 sp_type 조회.
    (astroquery Simbad 는 Windows numpy 오버플로 버그가 있어 우회)"""
    if os.path.exists(SIMBAD_CACHE):
        print(f"[SIMBAD] 캐시 사용: {SIMBAD_CACHE}")
        return pd.read_csv(SIMBAD_CACHE)
    import io
    import urllib.request
    import urllib.parse
    url = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
    frames = []
    names = [str(n).strip() for n in names if str(n).strip()]
    CH = 150
    print(f"[SIMBAD] {len(names)}개 식별자 조회 (청크 {CH})...")
    for s in range(0, len(names), CH):
        chunk = names[s:s + CH]
        inlist = ",".join("'" + n.replace("'", "''") + "'" for n in chunk)
        q = ("SELECT i.id AS name, b.main_id, b.sp_type "
             "FROM ident AS i JOIN basic AS b ON b.oid = i.oidref "
             f"WHERE i.id IN ({inlist})")
        data = urllib.parse.urlencode({
            'REQUEST': 'doQuery', 'LANG': 'ADQL',
            'FORMAT': 'csv', 'QUERY': q}).encode()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, data=data),
                        timeout=120) as resp:
                    frames.append(pd.read_csv(
                        io.BytesIO(resp.read())))
                break
            except Exception as e:
                print(f"  재시도 {attempt+1}/3: {type(e).__name__}")
                if attempt == 2:
                    raise
        print(f"  {min(s+CH, len(names))}/{len(names)}", flush=True)
    r = pd.concat(frames, ignore_index=True)
    r.columns = [c.lower() for c in r.columns]
    r.to_csv(SIMBAD_CACHE, index=False)
    print(f"[SIMBAD] sp_type 있음: {r['sp_type'].notna().sum()}/{len(r)}")
    return r


def mk_letter(sptype):
    """SIMBAD 분광형 문자열에서 주 온도형 추출 → 6클래스로 매핑.
    규칙: 대문자만 매칭('IIIb'의 b 오인 방지) / 탄소·S형 별은 제외 /
    Am 별 표기(kA5hF0mF3)는 수소선 타입(h 뒤 문자)을 온도형으로.
    예: 'K1.5III'→K, 'sdG0'→G, 'B9.5V'→OB, 'DA2.4'→WD, 'C-R4IIIb'→None"""
    if not isinstance(sptype, str) or not sptype.strip():
        return None
    s = sptype.strip()
    if re.match(r'^D[ABCOQZX]', s):
        return 'WD'                       # 백색왜성 (DA/DB 등)
    if re.match(r'^(C[-*0-9]|C$|S[0-9*]|S$)', s):
        return None                       # 탄소별·S형 (화학특이 — 비교 제외)
    m = re.search(r'h([OBAFGKM])', s)     # Am: 수소선 타입 우선
    if not m:
        m = re.search(r'[OBAFGKM]', s)    # 원문 대문자만
    if not m:
        return None
    c = m.group(1) if m.lastindex else m.group(0)
    return 'OB' if c in ('O', 'B') else c


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = pd.read_csv(MILES_GAIA)
    pred = pd.read_csv(PRED_CSV)
    pred['mid'] = pred['id'].str.extract(r's(\d+)').astype(float)
    df = pred.merge(cat, left_on='mid', right_on='MILES_ID', how='inner')

    name_col = 'simbad_name' if 'simbad_name' in df.columns else 'Name'
    sim = fetch_simbad_sptypes(df[name_col].fillna(df['Name']).tolist())
    sim = sim.drop_duplicates('name')
    df = df.merge(sim, left_on=name_col, right_on='name', how='left')

    df['simbad_cls'] = df['sp_type'].map(mk_letter)
    df['teff_cls'] = df['Teff'].map(
        lambda t: teff_to_class(t) if np.isfinite(t) else None)

    have = df[df['simbad_cls'].notna() &
              df['simbad_cls'].isin(CLASS_ORDER)].copy()
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    have['d_ai_simbad'] = (have['pred_label'].map(idx)
                           - have['simbad_cls'].map(idx)).abs()

    lines = []
    def log(s):
        print(s); lines.append(s)

    log(f"MILES {len(df)}개 중 SIMBAD 분광형 확보: {len(have)}개")
    agree = (have['pred_label'] == have['simbad_cls']).mean()
    adj = (have['d_ai_simbad'] <= 1).mean()
    log(f"AI ↔ SIMBAD 일치율: {100*agree:.1f}%  (±1클래스 {100*adj:.1f}%)")

    # Teff 라벨(균일한 물리량) 기준 심판
    both = have[have['teff_cls'].notna()].copy()
    mism = both[both['pred_label'] != both['simbad_cls']].copy()
    ai_side = mism[mism['pred_label'] == mism['teff_cls']]
    sb_side = mism[mism['simbad_cls'] == mism['teff_cls']]
    log(f"\n불일치 {len(mism)}개의 '심판'(문헌 Teff 구간과 비교):")
    log(f"  Teff 가 AI 편     : {len(ai_side)}개 "
        f"→ SIMBAD 라벨 재검토 후보")
    log(f"  Teff 가 SIMBAD 편 : {len(sb_side)}개 → AI 오분류")
    log(f"  Teff 가 제3의 값  : {len(mism)-len(ai_side)-len(sb_side)}개")

    # 재검토 후보: AI 확신 높음 + Teff 가 AI 편
    cand = ai_side[ai_side['pred_prob'] > 0.8].copy()
    cand = cand.sort_values('d_ai_simbad', ascending=False)
    cols = ['Name', 'sp_type', 'simbad_cls', 'pred_label', 'pred_prob',
            'Teff', 'teff_cls', 'Logg', 'lum', 'teff_K', 'logg']
    cand[cols].to_csv(os.path.join(OUT_DIR, "SIMBAD_재검토후보.csv"),
                      index=False, encoding='utf-8-sig')
    log(f"\n[재검토 후보] AI 확신>80% + Teff 일치: {len(cand)}개 "
        f"→ SIMBAD_재검토후보.csv")
    for _, r in cand.head(12).iterrows():
        log(f"  {str(r['Name']):16s} SIMBAD='{r['sp_type']}'({r['simbad_cls']})"
            f" → AI {r['pred_label']} ({100*r['pred_prob']:.0f}%)"
            f" | 문헌 Teff {r['Teff']:.0f}K={r['teff_cls']}"
            f" | AI Teff {r['teff_K']:.0f}K")

    # 백색왜성으로 SIMBAD 에 등록된 별 (있다면 광도계급 교차 확인)
    wd_sim = df[df['sp_type'].astype(str).str.match(r'^D[ABCOQZX]',
                                                    na=False)]
    if len(wd_sim):
        hit = (wd_sim['lum'] == 'wd').sum()
        log(f"\nSIMBAD 백색왜성(DA/DB형) {len(wd_sim)}개 중 "
            f"AI 도 백색왜성 판정: {hit}개")

    with open(os.path.join(OUT_DIR, "요약.txt"), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n저장: {OUT_DIR}/")


if __name__ == '__main__':
    main()
