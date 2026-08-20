# -*- coding: utf-8 -*-
"""
step1_v5.py — v5 전처리 실행기 (전처리 본체는 preprocess_core.py 를 import)

v3(step1_multi_survey_v3.py) 대비:
  1) 전처리·피처 로직이 이 파일에 없다 — preprocess_core.process_one 호출만.
     (학습·GUI 가 같은 모듈을 쓰므로 전처리 불일치가 원천적으로 불가능)
  2) 성간 소광 보정 복원 — master_catalog_v5.csv 의 av 컬럼(Gaia DR3
     azero_gspphot)을 spec_id 로 매핑해 전 서베이에 적용
     --noext 로 끄면 미보정 산출물 생성 (A/B 비교 실험용)
  3) 피처 39개 (EW 27 + FWHM 5 + 코어/날개 5 + 색지수 2)
  4) 출력: outputs_preprocessed_v5/ (미보정은 outputs_preprocessed_v5_noext/)

사용법 (v5 폴더에서):
  python step1_v5.py miles  --workers 4
  python step1_v5.py all    --workers 8
  python step1_v5.py lamost --workers 8 --noext   # 소광 미보정판
"""

import os
import re
import sys
import glob
import time
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

from preprocess_core import (process_one, vac_to_air, safe_arr,
                             N_PIX, WAVE_MIN, WAVE_MAX,
                             FEATURE_NAMES, LINE_NAMES, C_KMS)

BASE      = r'C:\Users\user\Desktop\최종'
RAW_ROOT  = os.path.join(BASE, '스펙트럼원본')
AV_CSV    = os.path.join(BASE, 'v5', 'master_catalog_v5.csv')


# ── star_id → spec_id 키 (build_catalog 규칙과 동일) ────────────────────
def spec_key(survey, star_id, fname):
    s = str(star_id).strip()
    if survey == 'mastar':
        return 'MASTAR-' + s
    if survey == 'lamost':
        return 'LAMOST-' + re.sub(r'\.0$', '', s)
    if survey == 'segue1':
        try:
            p, m, f = s.split('-')
            return f"SEGUE-{int(p)}-{int(m)}-{int(f)}"
        except Exception:
            return 'SEGUE-?'
    if survey == 'miles':
        m = re.match(r's(\d+)\.fits', str(fname))
        return f"MILES-{int(m.group(1)):04d}" if m else 'MILES-?'
    return '?'


def load_av_map():
    if not os.path.exists(AV_CSV):
        print(f"X {AV_CSV} 없음 — 먼저 python build_av_v5.py")
        sys.exit(1)
    df = pd.read_csv(AV_CSV, usecols=['spec_id', 'av'])
    av = pd.to_numeric(df['av'], errors='coerce')
    mp = dict(zip(df['spec_id'], av))
    n = int(av.notna().sum())
    print(f"[A_V] {n:,}/{len(df):,}개 확보 ({100*n/len(df):.1f}%) — "
          f"없는 별은 미보정 처리")
    return mp


# ── 서베이별 리더 (v3 검증본 그대로) ────────────────────────────────────

def iter_mastar(root, files):
    spec_path = files[0]
    with fits.open(spec_path, memmap=True) as hdul:
        d = hdul[1].data
        cols = hdul[1].columns.names
        helio_col = next((c for c in ['HELIOV', 'HELIO_V'] if c in cols), None)
        id_all = d['MANGAID']
        wave_all = d['WAVE']
        flux_all = d['FLUX']
        helio_all = d[helio_col] if helio_col else None
        shared_wave = None
        if wave_all.ndim == 1:
            shared_wave = vac_to_air(safe_arr(wave_all))
        elif wave_all.ndim > 1 and np.array_equal(wave_all[0], wave_all[-1]):
            shared_wave = vac_to_air(safe_arr(wave_all[0]))
        n = len(d)
        for i in range(n):
            try:
                sid = (id_all[i].decode() if isinstance(id_all[i], bytes)
                       else str(id_all[i])).strip()
                wave = (shared_wave if shared_wave is not None
                        else vac_to_air(safe_arr(wave_all[i])))
                flux = safe_arr(flux_all[i])
                rv = float(helio_all[i]) if helio_all is not None else None
                yield sid, f"row{i}", wave, flux, rv
            except Exception as e:
                yield None, f"row{i}", None, None, str(e)


def iter_segue(root, files):
    rv_map = {}
    gaia_csv = glob.glob(os.path.join(root, "**", "segue1_catalog_gaia.csv"),
                         recursive=True)
    if gaia_csv:
        cat = pd.read_csv(gaia_csv[0])
        rv_col = next((c for c in cat.columns
                       if c.lower() == "elodiervfinal"), None)
        if rv_col:
            for _, r in cat.iterrows():
                key = f"{int(r['plate']):04d}-{int(r['mjd'])}-{int(r['fiberid']):04d}"
                v = r[rv_col]
                rv_map[key] = float(v) if np.isfinite(v) else None

    name_re = re.compile(r"spec-(\d+)-(\d+)-(\d+)\.fits", re.IGNORECASE)
    for f in files:
        try:
            with fits.open(f, memmap=True) as hdul:
                d = hdul[1].data
                flux = safe_arr(d["flux"])
                wave = vac_to_air(10.0 ** safe_arr(d["loglam"]))
            m = name_re.search(os.path.basename(f))
            sid = (f"{int(m.group(1)):04d}-{m.group(2)}-{int(m.group(3)):04d}"
                   if m else os.path.basename(f))
            yield sid, os.path.basename(f), wave, flux, rv_map.get(sid)
        except Exception as e:
            yield None, os.path.basename(f), None, None, str(e)


def iter_miles(root, files):
    for f in files:
        try:
            with fits.open(f, memmap=True) as hdul:
                h = hdul[0].header
                flux = safe_arr(hdul[0].data).flatten()
                wave = h["CRVAL1"] + h["CDELT1"] * np.arange(len(flux))
            sid = str(h.get("OBJECT", os.path.basename(f))).strip()
            yield sid, os.path.basename(f), wave, flux, None
        except Exception as e:
            yield None, os.path.basename(f), None, None, str(e)


def iter_lamost(root, files):
    for f in files:
        try:
            with fits.open(f, memmap=True) as hdul:
                wave, flux = None, None
                if (len(hdul) > 1 and hasattr(hdul[1], "columns")
                        and hdul[1].columns):
                    names = hdul[1].columns.names
                    low = [c.lower() for c in names]
                    d = hdul[1].data
                    if "flux" in low:
                        flux = safe_arr(d[names[low.index("flux")]]).flatten()
                    if "wavelength" in low:
                        wave = safe_arr(
                            d[names[low.index("wavelength")]]).flatten()
                    elif "loglam" in low:
                        wave = 10.0 ** safe_arr(
                            d[names[low.index("loglam")]]).flatten()
                if flux is None and hdul[0].data is not None:
                    arr = safe_arr(hdul[0].data)
                    flux = arr[0] if arr.ndim > 1 else arr
                    h = hdul[0].header
                    if "COEFF0" in h and "COEFF1" in h:
                        wave = 10.0 ** (h["COEFF0"]
                                        + h["COEFF1"] * np.arange(len(flux)))
                z = hdul[0].header.get("Z")
                rv = z * C_KMS if z is not None and np.isfinite(z) else None
            if wave is None or flux is None:
                yield None, os.path.basename(f), None, None, "파장/플럭스 추출 실패"
                continue
            sid = os.path.basename(f).replace("spec_", "").replace(".fits", "")
            yield sid, os.path.basename(f), vac_to_air(wave), flux, rv
        except Exception as e:
            yield None, os.path.basename(f), None, None, str(e)


SURVEY_CONFIG = {
    "mastar": {"pattern": "mastar-goodspec*.fits", "folder_kw": "mastar",
               "iterator": iter_mastar},
    "segue1": {"pattern": "spec-*.fits", "folder_kw": "segue",
               "iterator": iter_segue},
    "miles":  {"pattern": "s*.fits", "folder_kw": "miles",
               "iterator": iter_miles},
    "lamost": {"pattern": "spec*.fits", "folder_kw": "lamost",
               "iterator": iter_lamost},
}


def find_files(root, cfg):
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if os.path.isdir(full) and cfg["folder_kw"] in d.lower():
            return sorted(glob.glob(
                os.path.join(full, "**", cfg["pattern"]), recursive=True))
    return []


def run_survey(survey, args, av_map):
    cfg = SURVEY_CONFIG[survey]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = find_files(RAW_ROOT, cfg)
    if not files:
        print(f"{survey}: 대상 파일 없음 ({RAW_ROOT})")
        return

    shard_idx, shard_n = map(int, args.shard.split("/"))
    if survey != "mastar":
        files = files[shard_idx::shard_n]
        total = len(files)
        print(f"{survey}: {total:,}개 파일")
    else:
        with fits.open(files[0], memmap=True) as h:
            total = int(h[1].header["NAXIS2"])
        print(f"mastar: 단일 파일 {total:,}행")

    use_ext = not args.noext

    def run_item(item):
        sid, fname, wave, flux, rv_or_err = item
        if wave is None:
            return sid, fname, f"read_fail:{rv_or_err}", None, None, np.nan
        rv = rv_or_err
        av = av_map.get(spec_key(survey, sid, fname)) if use_ext else None
        if av is not None and not np.isfinite(av):
            av = None
        try:
            res = process_one(wave, flux, rv, av)
        except Exception as e:
            return sid, fname, f"process_fail:{e}", None, None, np.nan
        if res is None:
            return sid, fname, "process_fail", None, None, np.nan
        return sid, fname, "ok", res[0], res[1], (av if av is not None
                                                  else np.nan)

    flux_rows, feat_rows, meta_rows = [], [], []
    n_ok, n_fail, row_count = 0, 0, 0
    t0 = time.time()

    def handle(sid, fname, status, f_out, feats, av_used):
        nonlocal n_ok, n_fail
        if status == "ok":
            meta_rows.append({"star_id": sid, "file": fname, "status": "ok",
                              "flux_row": n_ok, "av_used": av_used})
            flux_rows.append(f_out)
            feat_rows.append(feats)
            n_ok += 1
        else:
            meta_rows.append({"star_id": sid, "file": fname, "status": status,
                              "flux_row": -1, "av_used": np.nan})
            n_fail += 1
        done = n_ok + n_fail
        if done % 5000 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            left = max(total - done, 0) / max(rate, 1e-9)
            print(f"  진행 {done:,}/{total:,} ({rate:.0f}개/초, "
                  f"남은 예상 {left/60:.0f}분)", flush=True)

    it = cfg["iterator"](RAW_ROOT, files)
    if args.workers > 1 and survey != "mastar":
        from joblib import Parallel, delayed
        CHUNK = 2000
        pool = Parallel(n_jobs=args.workers, prefer="threads", batch_size=64)
        buf = []
        def flush(buf):
            if not buf:
                return
            for r in pool(delayed(run_item)(x) for x in buf):
                handle(*r)
            buf.clear()
        for item in it:
            buf.append(item)
            if len(buf) >= CHUNK:
                flush(buf)
        flush(buf)
    else:
        for item in it:
            row_count += 1
            if survey == "mastar" and (row_count - 1) % shard_n != shard_idx:
                continue
            handle(*run_item(item))

    tag = f"{survey}_shard{shard_idx}of{shard_n}"
    flux_arr = np.array(flux_rows, dtype=np.float32)
    feat_arr = np.array(feat_rows, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    n_meta_ok = int((meta_df["status"] == "ok").sum())
    assert flux_arr.shape[0] == n_meta_ok == n_ok, \
        f"정렬 오류: flux {flux_arr.shape[0]} vs meta ok {n_meta_ok}"
    fr = meta_df.loc[meta_df["status"] == "ok", "flux_row"].values
    assert np.array_equal(fr, np.arange(n_ok)), "flux_row 불연속!"

    np.save(os.path.join(outdir, f"{tag}_flux.npy"), flux_arr)
    np.save(os.path.join(outdir, f"{tag}_features.npy"), feat_arr)
    meta_df.to_csv(os.path.join(outdir, f"{tag}_meta.csv"),
                   index=False, encoding="utf-8-sig")

    n_av = int(np.isfinite(meta_df["av_used"]).sum())
    print(f"완료: 성공 {n_ok:,} / 실패 {n_fail:,}  "
          f"(소광 보정 적용 {n_av:,}개)")
    print(f"  {tag}_flux.npy ({n_ok}, {N_PIX})  "
          f"features ({n_ok}, {len(FEATURE_NAMES)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("survey", choices=list(SURVEY_CONFIG.keys()) + ["all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--noext", action="store_true",
                    help="소광 보정 끄기 (A/B 비교용)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = ("outputs_preprocessed_v5_noext" if args.noext
                       else "outputs_preprocessed_v5")

    av_map = {} if args.noext else load_av_map()
    if args.noext:
        print("!! --noext: 소광 미보정 모드")

    surveys = (list(SURVEY_CONFIG.keys()) if args.survey == "all"
               else [args.survey])
    for s in surveys:
        print(f"\n{'='*50}\n[{s}] 시작\n{'='*50}")
        run_survey(s, args, av_map)


if __name__ == "__main__":
    main()
