# -*- coding: utf-8 -*-
"""
prep_dataset_v5.py — v5 셰이드 + master_catalog_v5 → 학습 데이터셋

v4(prep_dataset_v4.py) 대비 변경:
  1) 6클래스 (OB/A/F/G/K/M) — 교과서 Teff 경계, preprocess_core 에서 import
  2) 광도계급 3분류 (사용자 결정 v5.1): 거성(logg<3.5) / 주계열(3.5~5.5) /
     백색왜성(>5.5). 콤팩트를 학습에 '포함' (LOGG_MAX=None) — 백색왜성이
     정식 클래스가 됐으므로. 표본 288개뿐이라 증강 필수 + 한계 문서화
  3) 증강 재설계 (train 전용, 그룹 유지):
     - O급(Teff>30000): ×5  — OB 클래스 내부 온도 회귀의 고온 꼬리 보존
     - 백색왜성(logg>5.5): ×5 — 표본 부족 완화 (함정 5번)
     - B급(10000<Teff<30000): ×2 유지
  4) 피처 39개 그대로 (물리 파라미터 입력 없음 — v4 함정 1번 준수)
  5) 파일명 v5_* / 출력 폴더 data/

실행 (v5 폴더에서, step1_v5.py 완료 후):
    python prep_dataset_v5.py
    python prep_dataset_v5.py --shard-dir outputs_preprocessed_v5_noext --out-dir data_noext
"""

import os
import re
import sys
import glob
import collections
import numpy as np
import pandas as pd

from preprocess_core import (teff_to_class, logg_to_lum, near_boundary,
                             CLASS_ORDER, N_PIX, N_FEATURES, O_TEFF)

BASE        = r'C:\Users\user\Desktop\최종'
SHARD_DIR   = os.path.join(BASE, 'v5', 'outputs_preprocessed_v5')
CATALOG_CSV = os.path.join(BASE, 'v5', 'master_catalog_v5.csv')
OUT_DIR     = os.path.join(BASE, 'v5', 'data')

SEED      = 42
TEST_SIZE = 0.15
CHUNK     = 8192

CLASS_CAP = {'OB': None, 'A': 15000,
             'F': 20000, 'G': 20000, 'K': 20000, 'M': 15000}
BOUNDARY_BUFFER_K = 75.0
LOGG_MAX = None             # v5.1: 백색왜성 포함 (광도계급 정식 클래스)

AUG_NOISE_RANGE = (0.005, 0.04)
AUG_TILT_MAX    = 0.05
AUG_SHIFT_PIX   = 2
AUG_FEAT_SD     = 0.01

TRUSTED_SOURCES = {'MASTAR', 'MILES'}


def aug_mults(ent):
    """train 각 행의 오프라인 증강 배수 벡터 (원본 포함).
    O급(Teff>30000) ×5 / B급(OB 나머지) ×2 / 백색왜성 ×5"""
    t = ent['teff_adopted'].values
    lab = ent['LABEL'].values
    lum = ent['LUM'].values
    m = np.ones(len(ent), dtype=int)
    m[lab == 'OB'] = 2
    m[(lab == 'OB') & (t > O_TEFF)] = 5
    m = np.maximum(m, np.where(lum == 'wd', 5, 1))
    return m


def make_keys(survey, meta):
    sid = meta['star_id'].astype(str).str.strip()
    if survey == 'mastar':
        return 'MASTAR-' + sid
    if survey == 'lamost':
        return 'LAMOST-' + sid.str.replace(r'\.0$', '', regex=True)
    if survey == 'segue1':
        def seg(s):
            try:
                p, m, f = s.split('-')
                return f"SEGUE-{int(p)}-{int(m)}-{int(f)}"
            except Exception:
                return "SEGUE-?"
        return sid.map(seg)
    if survey == 'miles':
        def mil(fn):
            m = re.match(r's(\d+)\.fits', str(fn))
            return f"MILES-{int(m.group(1)):04d}" if m else "MILES-?"
        return meta['file'].map(mil)
    raise ValueError(f"모르는 서베이: {survey}")


def collect_entries(cat, shard_dir):
    metas = sorted(glob.glob(os.path.join(shard_dir, "*_meta.csv")))
    if not metas:
        print(f"X {shard_dir}/ 에 셰이드 없음 — step1_v5.py 먼저")
        sys.exit(1)
    parts = []
    print(f"[셰이드] {len(metas)}개 ({shard_dir})")
    for mp in metas:
        tag = os.path.basename(mp)[:-len("_meta.csv")]
        survey = tag.split('_shard')[0]
        fluxp = os.path.join(shard_dir, f"{tag}_flux.npy")
        featp = os.path.join(shard_dir, f"{tag}_features.npy")
        if not (os.path.exists(fluxp) and os.path.exists(featp)):
            print(f"  !! {tag}: flux/features 없음 → 건너뜀"); continue
        meta = pd.read_csv(mp)
        ok = (meta['status'] == 'ok').values
        row_in_flux = meta['flux_row'].values.astype(np.int64)
        n_flux = np.load(fluxp, mmap_mode='r').shape[0]
        if ok.sum() != n_flux:
            print(f"  !! {tag}: meta ok {ok.sum():,} != flux {n_flux:,}")
            continue
        df = pd.DataFrame({'key': make_keys(survey, meta),
                           'flux_row': row_in_flux, 'tag': tag})[ok]
        df = df.merge(cat, left_on='key', right_on='spec_id', how='left')
        n_match = df['spec_id'].notna().sum()
        print(f"  {tag}: ok {ok.sum():>7,}  매칭 {n_match:>7,} "
              f"({100.0 * n_match / max(ok.sum(), 1):.1f}%)")
        parts.append(df[df['spec_id'].notna()])
    ent = pd.concat(parts, ignore_index=True)
    chk = ent[~ent['tag'].str.startswith('mastar')]
    dup = chk[chk.duplicated('key', keep=False)]
    if len(dup):
        print(f"X 중복 스펙트럼 {dup['key'].nunique():,}개"); sys.exit(1)
    return ent


def filter_and_cap(ent, rng):
    t = ent['teff_adopted'].values
    valid = np.isfinite(t) & (t > 1000) & (t < 60000)
    ent = ent[valid].copy()

    g = ent['logg_adopted'].values
    if LOGG_MAX is not None:
        compact = np.isfinite(g) & (g > LOGG_MAX)
        print(f"콤팩트(logg>{LOGG_MAX}) 제외: {compact.sum():,}개")
        ent = ent[~compact].copy()
    else:
        n_wd = int((np.isfinite(g) & (g > 5.5)).sum())
        print(f"백색왜성(logg>5.5) {n_wd:,}개 포함 — LUM 'wd' 클래스")

    t = ent['teff_adopted'].values
    buf = np.array([near_boundary(x, BOUNDARY_BUFFER_K) for x in t])
    print(f"경계 버퍼(±{BOUNDARY_BUFFER_K:.0f}K) 제외: {buf.sum():,}개")
    ent = ent[~buf].copy()

    ent['LABEL'] = [teff_to_class(x) for x in ent['teff_adopted'].values]
    ent['LUM']   = [logg_to_lum(x) for x in ent['logg_adopted'].values]

    snr = pd.to_numeric(ent.get('snr'), errors='coerce')
    score = snr.copy()
    score[ent['source'].isin(TRUSTED_SOURCES) & snr.isna()] = np.inf
    score[~ent['source'].isin(TRUSTED_SOURCES) & snr.isna()] = -np.inf
    ent['_snr_score'] = score.values

    picked = []
    print("\n[클래스별 개수 (캡: 서베이 층화 + SNR 순)]")
    for c in CLASS_ORDER:
        pool = ent[ent['LABEL'] == c]
        cap = CLASS_CAP.get(c)
        if cap is not None and len(pool) > cap:
            pool = pool.iloc[rng.permutation(len(pool))]
            counts = pool['source'].value_counts()
            quota = (counts / counts.sum() * cap).astype(int)
            rest = int(cap - quota.sum())
            for src in counts.index[:rest]:
                quota[src] += 1
            sel = []
            for src, q in quota.items():
                sub = pool[pool['source'] == src]
                sel.append(sub.sort_values('_snr_score', ascending=False,
                                           kind='stable').head(int(q)))
            pool = pd.concat(sel)
        srcs = collections.Counter(pool['source'])
        lums = collections.Counter(pool['LUM'])
        print(f"  {c}: {len(pool):>7,}  {dict(srcs)}")
        print(f"      광도: {dict(lums)}")
        picked.append(pool)
    return pd.concat(picked, ignore_index=True)


def build_aug_plan(ent_tr, rng):
    mults = aug_mults(ent_tr)
    rows = [(p, k) for p in np.where(mults > 1)[0]
            for k in range(1, mults[p])]
    if not rows:
        return None
    plan = pd.DataFrame(rows, columns=['orig_pos', 'aug_k'])
    cnt = collections.Counter(
        ent_tr['LABEL'].values[plan['orig_pos']])
    lum_cnt = collections.Counter(
        ent_tr['LUM'].values[plan['orig_pos']])
    print(f"\n[증강] train 복사본 {len(plan):,}개 "
          f"(클래스 {dict(cnt)} / 광도 {dict(lum_cnt)}) — test 원본만!")
    return plan


def perturb(f, rng):
    f = f.astype(np.float32).copy()
    s = int(rng.integers(-AUG_SHIFT_PIX, AUG_SHIFT_PIX + 1))
    if s != 0:
        f = np.roll(f, s)
        if s > 0: f[:s] = 1.0
        else:     f[s:] = 1.0
    a = float(rng.uniform(-AUG_TILT_MAX, AUG_TILT_MAX))
    f = f * (1.0 + a * np.linspace(-1.0, 1.0, len(f), dtype=np.float32))
    sig = float(rng.uniform(*AUG_NOISE_RANGE))
    f = f + rng.normal(0.0, sig, len(f)).astype(np.float32)
    return np.clip(f, 0.0, 5.0)


def save_split(tag, ent_part, feats_part, out_dir, shard_dir,
               aug_plan=None, rng=None):
    n_orig = len(ent_part)
    n_aug = 0 if aug_plan is None else len(aug_plan)
    n = n_orig + n_aug

    outp = os.path.join(out_dir, f"v5_{tag}_flux.npy")
    out = np.lib.format.open_memmap(outp, mode='w+', dtype=np.float32,
                                    shape=(n, N_PIX))
    pos = 0
    order = []
    ent_part = ent_part.reset_index(drop=True)
    for stag, grp in ent_part.groupby('tag', sort=True):
        grp = grp.sort_values('flux_row')
        rows = grp['flux_row'].values
        mm = np.load(os.path.join(shard_dir, f"{stag}_flux.npy"),
                     mmap_mode='r')
        for s in range(0, len(rows), CHUNK):
            r = rows[s:s + CHUNK]
            out[pos:pos + len(r)] = mm[r]
            pos += len(r)
        order.extend(grp.index.tolist())
    assert pos == n_orig

    aug_meta = []
    if n_aug:
        by_tag = {}
        for _, r in aug_plan.iterrows():
            src = ent_part.iloc[int(r['orig_pos'])]
            by_tag.setdefault(src['tag'], []).append(
                (int(src['flux_row']), int(r['orig_pos']), int(r['aug_k'])))
        for stag, items in sorted(by_tag.items()):
            mm = np.load(os.path.join(shard_dir, f"{stag}_flux.npy"),
                         mmap_mode='r')
            for flux_row, orig_pos, aug_k in items:
                out[pos] = perturb(np.asarray(mm[flux_row]), rng)
                aug_meta.append((orig_pos, aug_k))
                pos += 1
    out.flush(); del out
    assert pos == n

    feat_orig = feats_part[np.array(order, dtype=int)]
    if n_aug:
        idx = np.array([m[0] for m in aug_meta], dtype=int)
        feat_aug = feats_part[idx] + rng.normal(
            0.0, AUG_FEAT_SD, (n_aug, feats_part.shape[1])
        ).astype(np.float32)
        feat_all_out = np.vstack([feat_orig, feat_aug]).astype(np.float32)
    else:
        feat_all_out = feat_orig.astype(np.float32)
    np.save(os.path.join(out_dir, f"v5_{tag}_features.npy"), feat_all_out)

    ordered = ent_part.iloc[order]
    def frame(src_rows, star_ids, augs):
        return pd.DataFrame({
            'STAR_ID': star_ids,
            'SOURCE':  src_rows['source'].values,
            'TEFF':    src_rows['teff_adopted'].values,
            'LOGG':    src_rows['logg_adopted'].values,
            'FEH':     src_rows['feh_adopted'].values,
            'AV':      pd.to_numeric(src_rows.get('av'),
                                     errors='coerce').values,
            'LABEL':   src_rows['LABEL'].values,
            'LUM':     src_rows['LUM'].values,
            'SNR':     src_rows['_snr_score'].replace(
                           [np.inf, -np.inf], np.nan).values,
            'GROUP':   src_rows['group_id'].values,
            'AUG':     augs,
        })
    frames = [frame(ordered, ordered['spec_id'].values, 0)]
    if n_aug:
        srcs = ent_part.iloc[[m[0] for m in aug_meta]]
        frames.append(frame(
            srcs,
            [f"{sid}#aug{k}" for sid, (_, k)
             in zip(srcs['spec_id'].values, aug_meta)],
            [k for _, k in aug_meta]))
    lab = pd.concat(frames, ignore_index=True)
    lab.to_csv(os.path.join(out_dir, f"v5_{tag}_labels.csv"), index=False)

    d = collections.Counter(lab['LABEL'])
    lum_d = collections.Counter(lab['LUM'])
    aug_note = f" (원본 {n_orig:,} + 증강 {n_aug:,})" if n_aug else ""
    print(f"  [{tag}] {n:,}개{aug_note}")
    print(f"    분광형: " + "  ".join(
        f"{c}:{d.get(c, 0):,}" for c in CLASS_ORDER))
    print(f"    광도: {dict(lum_d)}")


def main():
    import argparse
    from sklearn.model_selection import GroupShuffleSplit
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocap", action="store_true")
    ap.add_argument("--noaug", action="store_true")
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    global CLASS_CAP
    if args.nocap:
        CLASS_CAP = {c: None for c in CLASS_ORDER}
    rng = np.random.default_rng(SEED)

    print("=" * 64)
    print("v5 데이터셋 — 6클래스(OB~M, 교과서 경계) / 광도 거성·주계열·백색왜성 / O급·WD 증강")
    print("=" * 64)

    cat = pd.read_csv(CATALOG_CSV)
    need = ['spec_id', 'source', 'group_id',
            'teff_adopted', 'logg_adopted', 'feh_adopted']
    for opt in ['snr', 'av']:
        if opt in cat.columns:
            need.append(opt)
    cat = cat[need].drop_duplicates('spec_id')
    print(f"[카탈로그] {len(cat):,}행")

    ent = collect_entries(cat, args.shard_dir)
    # 그룹 ID 타입 통일 (기존=숫자, M보강 신규=문자열 혼재 → 정렬 오류 방지)
    ent['group_id'] = ent['group_id'].astype(str)
    ent = filter_and_cap(ent, rng)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SEED)
    idx_tr, idx_te = next(gss.split(np.arange(len(ent)),
                                    groups=ent['group_id'].values))
    assert not (set(ent['group_id'].values[idx_tr])
                & set(ent['group_id'].values[idx_te]))
    print(f"\ntrain {len(idx_tr):,} / test {len(idx_te):,} (그룹 단위)")

    feat_parts = {}
    for stag in ent['tag'].unique():
        feat_parts[stag] = np.load(
            os.path.join(args.shard_dir, f"{stag}_features.npy"))
    n_feat = next(iter(feat_parts.values())).shape[1]
    assert n_feat == N_FEATURES, f"피처 수 불일치: {n_feat} != {N_FEATURES}"
    feat_all = np.empty((len(ent), n_feat), dtype=np.float32)
    for stag, grp in ent.groupby('tag'):
        feat_all[grp.index.values] = feat_parts[stag][grp['flux_row'].values]
    del feat_parts

    os.makedirs(args.out_dir, exist_ok=True)
    tr = ent.iloc[idx_tr]
    logteff = np.log10(tr['teff_adopted'].values)
    lt_mu, lt_sd = float(logteff.mean()), float(logteff.std() + 1e-8)
    np.save(os.path.join(args.out_dir, "teff_norm_v5.npy"),
            np.array([lt_mu, lt_sd], dtype=np.float64))
    g = tr['logg_adopted'].values
    ok = np.isfinite(g)
    lg_mu = float(np.nanmean(g[ok])); lg_sd = float(np.nanstd(g[ok]) + 1e-8)
    np.save(os.path.join(args.out_dir, "logg_norm_v5.npy"),
            np.array([lg_mu, lg_sd], dtype=np.float64))
    print(f"정규화(train만): logTeff mu={lt_mu:.4f} sd={lt_sd:.4f} | "
          f"logg mu={lg_mu:.2f} sd={lg_sd:.2f}")

    # 피처 정규화 통계 (train 원본 기준) — MLP 입력 표준화용
    f_mu = feat_all[idx_tr].mean(axis=0)
    f_sd = feat_all[idx_tr].std(axis=0) + 1e-8
    np.save(os.path.join(args.out_dir, "feat_norm_v5.npy"),
            np.vstack([f_mu, f_sd]).astype(np.float64))

    ent_tr = ent.iloc[idx_tr].reset_index(drop=True)
    ent_te = ent.iloc[idx_te].reset_index(drop=True)
    aug_plan = None if args.noaug else build_aug_plan(ent_tr, rng)

    print("\n저장...")
    save_split("train", ent_tr, feat_all[idx_tr], args.out_dir,
               args.shard_dir, aug_plan=aug_plan, rng=rng)
    save_split("test", ent_te, feat_all[idx_te], args.out_dir,
               args.shard_dir)
    print(f"\n완료 → {args.out_dir}/  다음: python train_v5.py")


if __name__ == "__main__":
    main()
