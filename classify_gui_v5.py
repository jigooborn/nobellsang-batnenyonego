# -*- coding: utf-8 -*-
"""
classify_gui_v5.py — v5 분류 GUI (6클래스 OB~M + 광도계급 + 원소선 표시)

v2 GUI 계승 + v5 변경:
  - 전처리·피처: preprocess_core.process_one 을 import (학습과 100% 동일 —
    이 파일 안에 전처리 복사본이 없음. 함정 3번 원천 차단)
  - 모델: models/resnet_v5.pth + mlp_v5.pth (6분류 + Teff + logg)
  - 입력 힌트 없음: logg/[Fe/H] 입력칸 제거 (v5 는 스펙트럼만 사용)
  - 결과 표시: "OB형 (약 21,000 K) · 거성" — 분광형 + 내부 온도 + 광도계급
    OB 이면서 Teff>30,000K + He II 검출 시 "O급 가능성" 표시
  - ★ 원소 흡수선 표시: 스펙트럼 위에 27개 선 위치·한글 이름 표시,
    예측 분광형의 판정 근거가 되는 선은 굵게 하이라이트
  - 🖼 그림 저장: 현재 스펙트럼 + 흡수선 라벨 PNG 내보내기 (보고서 그림용)

실행 (v5 폴더): python classify_gui_v5.py [파일.fits] [--batch out.csv]
"""

import os
import re
import sys
import glob
import argparse

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch 미설치. 실행: pip install torch")
    sys.exit(1)

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

import warnings
warnings.filterwarnings("ignore")

from preprocess_core import (process_one, vac_to_air, safe_arr,
                             TARGET_WAVE, N_PIX, WAVE_MIN, WAVE_MAX, C_KMS,
                             CLASS_ORDER, N_CLASSES, N_FEATURES,
                             FEATURE_NAMES, LINES_V5, O_TEFF,
                             LOGG_GIANT, LOGG_COMPACT, LUM_ORDER, LUM_KO,
                             estimate_rv, estimate_av)
from train_v5 import SpectralResNetV5, FeatureMLPV5, DEVICE

CLASS_KO = {'OB': 'OB형(청색·고온)', 'A': 'A형(흰색)', 'F': 'F형(황백색)',
            'G': 'G형(노란색·태양형)', 'K': 'K형(주황색)', 'M': 'M형(붉은색)'}

# 판정 배너용 클래스 색 (별의 실제 색감을 딴 파스텔 배경)
CLASS_BG = {'OB': '#cdd9ff', 'A': '#eef1fa', 'F': '#fdf6dd',
            'G': '#ffe9a8', 'K': '#ffd9ad', 'M': '#ffc4b0'}


def _sens_classes(sens):
    """'OB~A', 'F~K', 'M', 'O급' 같은 민감 분광형 문자열 → 클래스 집합"""
    s = sens.replace('급', '').replace('이른 ', '').strip()
    if '~' in s:
        a, b = s.split('~')
        a = 'OB' if a == 'O' else a
        b = 'OB' if b == 'O' else b
        try:
            i, j = CLASS_ORDER.index(a), CLASS_ORDER.index(b)
            return set(CLASS_ORDER[min(i, j):max(i, j) + 1])
        except ValueError:
            return set()
    s = 'OB' if s in ('O', 'B', 'OB') else s
    return {s} if s in CLASS_ORDER else set()


LINE_SENS = {l[0]: _sens_classes(l[6]) for l in LINES_V5}


# 광도계급은 모델의 전용 분류 헤드(거성/주계열/백색왜성)로 판정.
# 예측 logg 는 물리량 표시·교차 확인용으로만 사용.


def simbad_lookup(ra_deg, dec_deg, radius_arcsec=3.0):
    """좌표 원뿔 검색으로 SIMBAD 등록 정보 조회.
    반환: dict(main_id, sp_type, otype, sep_arcsec) 또는 None."""
    import io as _io
    import urllib.request
    import urllib.parse
    q = ("SELECT TOP 3 main_id, sp_type, otype, "
         f"DISTANCE(POINT('ICRS', ra, dec), "
         f"POINT('ICRS', {ra_deg:.6f}, {dec_deg:.6f})) AS d "
         "FROM basic "
         f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, "
         f"{radius_arcsec/3600.0:.7f}))=1 "
         "ORDER BY d")
    data = urllib.parse.urlencode({
        'REQUEST': 'doQuery', 'LANG': 'ADQL',
        'FORMAT': 'csv', 'QUERY': q}).encode()
    url = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
    try:
        import pandas as pd
        with urllib.request.urlopen(
                urllib.request.Request(url, data=data), timeout=20) as r:
            df = pd.read_csv(_io.BytesIO(r.read()))
    except Exception:
        return None
    if not len(df):
        return {'main_id': None}
    row = df.iloc[0]
    return {'main_id': str(row['main_id']),
            'sp_type': (str(row['sp_type'])
                        if isinstance(row['sp_type'], str) else ''),
            'otype': str(row.get('otype', '')),
            'sep_arcsec': float(row['d']) * 3600.0}


# ══════════════════════════════════════════════════════════════════════
# 예측기
# ══════════════════════════════════════════════════════════════════════
class PredictorV5:
    def __init__(self, model_dir=None):
        here = os.path.dirname(os.path.abspath(__file__))
        if model_dir is None:
            for cand in [os.path.join(here, "models"), "models"]:
                if os.path.exists(os.path.join(cand, "resnet_v5.pth")):
                    model_dir = cand
                    break
        if model_dir is None:
            raise FileNotFoundError("models/resnet_v5.pth 를 찾을 수 없습니다. "
                                    "train_v5.py 를 먼저 실행하세요.")
        data_dir = None
        for cand in [os.path.join(here, "data"), "data"]:
            if os.path.exists(os.path.join(cand, "teff_norm_v5.npy")):
                data_dir = cand
                break
        if data_dir is None:
            raise FileNotFoundError("data/teff_norm_v5.npy 가 없습니다.")

        self.lt_mu, self.lt_sd = [float(x) for x in
                                  np.load(os.path.join(data_dir,
                                                       "teff_norm_v5.npy"))]
        self.lg_mu, self.lg_sd = [float(x) for x in
                                  np.load(os.path.join(data_dir,
                                                       "logg_norm_v5.npy"))]
        fstats = np.load(os.path.join(data_dir, "feat_norm_v5.npy"))
        self.f_mu = fstats[0].astype(np.float32)
        self.f_sd = fstats[1].astype(np.float32)

        self.cnn = SpectralResNetV5().to(DEVICE)
        self.cnn.load_state_dict(torch.load(
            os.path.join(model_dir, "resnet_v5.pth"),
            map_location=DEVICE, weights_only=True))
        self.mlp = FeatureMLPV5().to(DEVICE)
        self.mlp.load_state_dict(torch.load(
            os.path.join(model_dir, "mlp_v5.pth"),
            map_location=DEVICE, weights_only=True))
        self.cnn.eval()
        self.mlp.eval()
        self.model_dir = model_dir

        # A_V 자동 추정용 보정 곡선 (무소광 학습 표본의 Teff-기울기 관계)
        self.slope_calib = None
        p = os.path.join(data_dir, "slope_calib_v5.npy")
        if os.path.exists(p):
            self.slope_calib = np.load(p)

    def predict(self, wave, flux, rv_kms=None, av=None):
        res = process_one(wave, flux, rv_kms, av)
        if res is None:
            return None
        flux_out, feats = res
        feat_std = (feats - self.f_mu) / self.f_sd

        fb = torch.from_numpy(flux_out[None, :]).float().to(DEVICE)
        xb = torch.from_numpy(feat_std[None, :]).float().to(DEVICE)
        with torch.no_grad():
            lc, tc, gc, uc = self.cnn(fb, xb)
            lm, tm, gm, um = self.mlp(fb, xb)
            pc = torch.softmax(lc, 1).cpu().numpy()[0]
            pm = torch.softmax(lm, 1).cpu().numpy()[0]
            pu = ((torch.softmax(uc, 1) + torch.softmax(um, 1)) / 2
                  ).cpu().numpy()[0]
            tc = float(tc.cpu().numpy()[0]); tm = float(tm.cpu().numpy()[0])
            gc = float(gc.cpu().numpy()[0]); gm = float(gm.cpu().numpy()[0])
        pe = (pc + pm) / 2
        denorm_t = lambda t: 10.0 ** (t * self.lt_sd + self.lt_mu)
        logg = ((gc + gm) / 2) * self.lg_sd + self.lg_mu
        teff = denorm_t((tc + tm) / 2)
        lum = LUM_ORDER[int(pu.argmax())]

        ews = dict(zip([l[0] for l in LINES_V5],
                       feats[:len(LINES_V5)].tolist()))
        pred = CLASS_ORDER[int(pe.argmax())]
        # O급 참고 판정: OB 인데 온도가 O 경계 이상 + He II 흡수 존재
        o_hint = (pred == 'OB' and teff > O_TEFF
                  and (ews.get('He4542', 0) > 0.3
                       or ews.get('He4686', 0) > 0.3))
        return {
            'flux_norm': flux_out, 'features': feats, 'ews': ews,
            'probs_cnn': pc, 'probs_mlp': pm, 'probs_ens': pe,
            'pred_cnn': CLASS_ORDER[int(pc.argmax())],
            'pred_mlp': CLASS_ORDER[int(pm.argmax())],
            'pred_ens': pred,
            'prob_cnn': float(pc.max()), 'prob_mlp': float(pm.max()),
            'prob_ens': float(pe.max()),
            'teff_cnn': denorm_t(tc), 'teff_mlp': denorm_t(tm),
            'teff_ens': teff,
            'logg': logg, 'lum': lum, 'lum_probs': pu,
            'lum_prob': float(pu.max()),
            'o_hint': o_hint,
        }

    def predict_auto(self, wave, flux, rv=None, av=None):
        """RV·A_V 를 스펙트럼에서 자동 추정하는 완전 자동 파이프라인.
        rv/av 를 주면 그 값을 우선 사용 (av=0 은 '보정 끔' 의미).
        반환 dict 에 rv_used/rv_auto/av_used/av_auto 추가."""
        rv_auto = av_auto = False
        if rv is None:
            rv = estimate_rv(wave, flux)
            rv_auto = rv is not None
        r = self.predict(wave, flux, rv, av)
        if r is None:
            return None
        if av is None and self.slope_calib is not None:
            # 안전장치: 추정 상한 1.5등급 (그 이상은 플럭스 보정 불량이나
            # 저온별 축퇴일 가능성이 커서 폭주 위험 — XSL 실측으로 확인),
            # 보정 후 분류 신뢰도가 떨어지면 보정을 되돌림
            slope_obs = float(r['features'][-2])   # 원 스펙트럼의 색지수
            av_est = estimate_av(r['teff_ens'], slope_obs, self.slope_calib)
            for _ in range(2):                     # 최대 2회 반복 수렴
                if av_est is None or av_est < 0.05 or av_est > 1.5:
                    break
                r2 = self.predict(wave, flux, rv, av_est)
                if r2 is None or r2['prob_ens'] < r['prob_ens'] - 0.05:
                    break                          # 신뢰도 악화 → 미보정 유지
                r, av_auto, av = r2, True, av_est
                new_est = estimate_av(r['teff_ens'], slope_obs,
                                      self.slope_calib)
                if new_est is None or abs(new_est - av_est) < 0.1:
                    break
                av_est = new_est
        r['rv_used'], r['rv_auto'] = rv, rv_auto
        r['av_used'], r['av_auto'] = av, av_auto
        return r


# ══════════════════════════════════════════════════════════════════════
# 스펙트럼 로더 (v2 검증본 그대로)
# ══════════════════════════════════════════════════════════════════════
class SpectrumSource:
    """파일 하나 = 스펙트럼 1개 이상. get(i) → (id, wave, flux, rv, av)"""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.kind = "unknown"
        self._hdul = None
        self.n = 0
        self.ids = []
        self._sub = None
        self._detect()

    def close(self):
        if self._sub is not None:
            self._sub.close()
            self._sub = None
        if self._hdul is not None:
            try:
                self._hdul.close()
            except Exception:
                pass

    @staticmethod
    def _id_from_name(fname):
        base = os.path.basename(fname)
        m = re.search(r"spec[_-]?(\d+)\.fits", base, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"spec-(\d+)-(\d+)-(\d+)", base)
        if m:
            return f"{int(m.group(1)):04d}-{m.group(2)}-{int(m.group(3)):04d}"
        return os.path.splitext(base)[0]

    def _detect(self):
        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, "*.fits")) +
                           glob.glob(os.path.join(self.path, "*.fit")))
            if not files:
                files = sorted(glob.glob(
                    os.path.join(self.path, "**", "*.fit*"), recursive=True))
            if not files:
                raise RuntimeError("폴더 안에 fits 파일이 없습니다.")
            self.kind = "folder"
            self._files = files
            self.n = len(files)
            self.ids = [self._id_from_name(f) for f in files]
            return

        low = self.name.lower()
        ext = os.path.splitext(low)[1]
        if ext in (".csv", ".txt", ".dat", ".tsv"):
            self.kind = "table"
            self.n = 1
            self.ids = [self.name]
            return
        if ext == ".npy":
            arr = np.load(self.path)
            self._npy = np.atleast_2d(arr)
            self.kind = "npy"
            self.n = self._npy.shape[0]
            self.ids = [f"row{i}" for i in range(self.n)]
            return
        if not HAS_ASTROPY:
            raise RuntimeError("FITS 를 읽으려면 astropy 가 필요합니다.")

        self._hdul = fits.open(self.path, memmap=True)
        hdul = self._hdul

        if len(hdul) > 1 and hasattr(hdul[1], "columns") and hdul[1].columns \
                and "MANGAID" in hdul[1].columns.names:
            self.kind = "mastar"
            self.n = len(hdul[1].data)
            self.ids = [(x.decode() if isinstance(x, bytes) else str(x)).strip()
                        for x in hdul[1].data["MANGAID"]]
            return
        if len(hdul) > 1 and hasattr(hdul[1], "columns") and hdul[1].columns:
            lowcols = [c.lower() for c in hdul[1].columns.names]
            if "loglam" in lowcols and "flux" in lowcols:
                self.kind = "sdss"
                self.n = 1
                m = re.search(r"spec-(\d+)-(\d+)-(\d+)", self.name)
                self.ids = [f"{int(m.group(1)):04d}-{m.group(2)}-"
                            f"{int(m.group(3)):04d}" if m else self.name]
                return
            if "flux" in lowcols and "wavelength" in lowcols:
                self.kind = "lamost"
                self.n = 1
                self.ids = [self.name.replace("spec_", "").replace(".fits", "")]
                return
        h0 = hdul[0].header
        if hdul[0].data is not None and "COEFF0" in h0 and "COEFF1" in h0:
            self.kind = "lamost"
            self.n = 1
            self.ids = [self.name.replace("spec_", "").replace(".fits", "")]
            return
        if hdul[0].data is not None and "CRVAL1" in h0 and \
                ("CDELT1" in h0 or "CD1_1" in h0):
            self.kind = "miles"
            self.n = 1
            self.ids = [str(h0.get("OBJECT", self.name)).strip()]
            return

        W_CAND = ["wavelength", "wave", "lambda", "wav", "awav", "loglam"]
        F_CAND = ["flux", "flux_density", "spec", "spectrum",
                  "intensity", "counts", "sci"]
        for k, hdu in enumerate(hdul):
            if not (hasattr(hdu, "columns") and hdu.columns):
                continue
            names = hdu.columns.names
            lowc = [c.lower() for c in names]
            wc = next((c for c in W_CAND if c in lowc), None)
            fc = next((c for c in F_CAND if c in lowc), None)
            if wc and fc:
                self.kind = "generic"
                self._ghdu, self._glog = k, (wc == "loglam")
                self._gwcol = names[lowc.index(wc)]
                self._gfcol = names[lowc.index(fc)]
                self.n = 1
                self.ids = [os.path.splitext(self.name)[0]]
                return

        info = []
        for k, hdu in enumerate(hdul):
            if hasattr(hdu, "columns") and hdu.columns:
                info.append(f"HDU{k}=테이블{list(hdu.columns.names)[:8]}")
            elif hdu.data is not None:
                info.append(f"HDU{k}=배열{getattr(hdu.data, 'shape', '?')}")
            else:
                info.append(f"HDU{k}=헤더만")
        raise RuntimeError(
            "지원하지 않는 FITS 형식입니다.\n\n파일 구조: " + " | ".join(info)
            + "\n\n이 메시지를 캡처해서 보내주시면 형식 지원을 추가하겠습니다.")

    def get(self, i):
        if self.kind == "folder":
            f = self._files[i]
            if self._sub is None or self._sub.path != f:
                if self._sub is not None:
                    self._sub.close()
                self._sub = SpectrumSource(f)
            sid, wave, flux, rv, av = self._sub.get(0)
            return self.ids[i], wave, flux, rv, av

        if self.kind == "table":
            import pandas as pd
            df = pd.read_csv(self.path, sep=None, engine="python",
                             comment="#", header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            wave = df.iloc[:, 0].values.astype(float)
            flux = df.iloc[:, 1].values.astype(float)
            return self.name, wave, flux, None, None

        if self.kind == "npy":
            f = self._npy[i].astype(float)
            if len(f) != N_PIX:
                raise RuntimeError(f"npy 는 {N_PIX}픽셀 정규화 flux 여야 합니다.")
            return f"{self.name}[{i}]", TARGET_WAVE.copy(), f, None, None

        hdul = self._hdul

        if self.kind == "mastar":
            d = hdul[1].data
            cols = hdul[1].columns.names
            sid = str(d["MANGAID"][i]).strip()
            wave = safe_arr(d["WAVE"][i] if d["WAVE"].ndim > 1 else d["WAVE"])
            flux = safe_arr(d["FLUX"][i])
            helio_col = next((c for c in ["HELIOV", "HELIO_V"]
                              if c in cols), None)
            rv = float(d[helio_col][i]) if helio_col else None
            return sid, vac_to_air(wave), flux, rv, None

        if self.kind == "sdss":
            d = hdul[1].data
            flux = safe_arr(d["flux"])
            wave = vac_to_air(10.0 ** safe_arr(d["loglam"]))
            m = re.search(r"spec-(\d+)-(\d+)-(\d+)", self.name)
            sid = (f"{int(m.group(1)):04d}-{m.group(2)}-{int(m.group(3)):04d}"
                   if m else self.name)
            rv = None
            try:
                z = float(hdul[2].data["Z"][0])
                if np.isfinite(z):
                    rv = z * C_KMS
            except Exception:
                pass
            return sid, wave, flux, rv, None

        if self.kind == "lamost":
            wave, flux = None, None
            if len(hdul) > 1 and hasattr(hdul[1], "columns") and hdul[1].columns:
                names = hdul[1].columns.names
                lowc = [c.lower() for c in names]
                d = hdul[1].data
                if "flux" in lowc:
                    flux = safe_arr(d[names[lowc.index("flux")]]).flatten()
                if "wavelength" in lowc:
                    wave = safe_arr(
                        d[names[lowc.index("wavelength")]]).flatten()
                elif "loglam" in lowc:
                    wave = 10.0 ** safe_arr(
                        d[names[lowc.index("loglam")]]).flatten()
            if flux is None and hdul[0].data is not None:
                arr = safe_arr(hdul[0].data)
                flux = arr[0] if arr.ndim > 1 else arr
                h = hdul[0].header
                if "COEFF0" in h and "COEFF1" in h:
                    wave = 10.0 ** (h["COEFF0"]
                                    + h["COEFF1"] * np.arange(len(flux)))
            if wave is None or flux is None:
                raise RuntimeError("LAMOST 파장/플럭스 추출 실패")
            z = hdul[0].header.get("Z")
            rv = z * C_KMS if z is not None and np.isfinite(z) else None
            sid = self.name.replace("spec_", "").replace(".fits", "")
            return sid, vac_to_air(wave), flux, rv, None

        if self.kind == "generic":
            d = hdul[self._ghdu].data
            wave = safe_arr(d[self._gwcol]).flatten()
            if self._glog:
                wave = 10.0 ** wave
            flux = safe_arr(d[self._gfcol]).flatten()
            return self.ids[0], wave, flux, None, None

        if self.kind == "miles":
            h = hdul[0].header
            flux = safe_arr(hdul[0].data).flatten()
            step = h.get("CDELT1", h.get("CD1_1"))
            wave = h["CRVAL1"] + step * np.arange(len(flux))
            if str(h.get("CTYPE1", "")).upper().startswith("LOG") or \
                    h.get("DC-FLAG") == 1:
                wave = 10.0 ** wave
            sid = str(h.get("OBJECT", self.name)).strip()
            return sid, wave, flux, None, None

        raise RuntimeError("unreachable")

    def get_radec(self, i):
        """스펙트럼 i 의 적경/적위(deg). 없으면 (None, None)."""
        try:
            if self.kind == "folder":
                self.get(i)                     # _sub 준비
                return self._sub.get_radec(0) if self._sub else (None, None)
            if self.kind == "mastar":
                d = self._hdul[1].data
                cols = self._hdul[1].columns.names
                rc = next((c for c in ['OBJRA', 'RA', 'IFURA']
                           if c in cols), None)
                dc = next((c for c in ['OBJDEC', 'DEC', 'IFUDEC']
                           if c in cols), None)
                if rc and dc:
                    return float(d[rc][i]), float(d[dc][i])
                return None, None
            if self._hdul is not None:
                h = self._hdul[0].header
                for rk, dk in [('PLUG_RA', 'PLUG_DEC'), ('RA', 'DEC'),
                               ('OBJRA', 'OBJDEC'), ('RA_OBJ', 'DEC_OBJ')]:
                    if rk in h and dk in h:
                        ra, dec = h[rk], h[dk]
                        if isinstance(ra, str):    # 육십분법 문자열
                            from astropy.coordinates import SkyCoord
                            import astropy.units as u
                            c = SkyCoord(f"{ra} {dec}",
                                         unit=(u.hourangle, u.deg))
                            return float(c.ra.deg), float(c.dec.deg)
                        return float(ra), float(dec)
        except Exception:
            pass
        return None, None


# ══════════════════════════════════════════════════════════════════════
# 일괄 분류
# ══════════════════════════════════════════════════════════════════════
def batch_classify(pred, src, out_csv, progress=None):
    import csv as _csv
    rows = []
    for i in range(src.n):
        try:
            sid, wave, flux, rv, av = src.get(i)
            r = pred.predict_auto(wave, flux, rv, av)
            if r is None:
                rows.append({"file": src.name, "id": sid,
                             "error": "전처리 실패"})
            else:
                rv_u = r.get('rv_used')
                av_u = r.get('av_used')
                row = {"file": src.name, "id": sid,
                       "rv_kms": f"{rv_u:.1f}" if rv_u is not None else "",
                       "rv_src": ("auto" if r.get('rv_auto') else
                                  ("file" if rv is not None else "")),
                       "av": f"{av_u:.2f}" if av_u is not None else "",
                       "av_src": "auto" if r.get('av_auto') else "",
                       "pred_label": r['pred_ens'],
                       "pred_prob": f"{r['prob_ens']:.4f}",
                       "teff_K": f"{r['teff_ens']:.0f}",
                       "logg": f"{r['logg']:.2f}",
                       "lum": r['lum'],
                       "o_hint": "O급?" if r['o_hint'] else "",
                       "cnn_label": r['pred_cnn'],
                       "mlp_label": r['pred_mlp'],
                       "error": ""}
                for c, p in zip(CLASS_ORDER, r['probs_ens']):
                    row[f"P_{c}"] = f"{p:.4f}"
                rows.append(row)
        except Exception as e:
            rows.append({"file": src.name, "id": f"row{i}", "error": str(e)})
        if progress:
            progress(i + 1, src.n)

    keys = ["file", "id", "rv_kms", "rv_src", "av", "av_src",
            "pred_label", "pred_prob", "teff_K",
            "logg", "lum", "o_hint", "cnn_label", "mlp_label"] + \
           [f"P_{c}" for c in CLASS_ORDER] + ["error"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ══════════════════════════════════════════════════════════════════════
# 스펙트럼 + 원소선 그리기 (GUI/그림 저장 공용)
# ══════════════════════════════════════════════════════════════════════
def draw_spectrum(ax, result, sid, show_lines=True, highlight=True):
    """정규화 스펙트럼 + 27개 원소선 라벨. 예측 클래스의 근거 선은 강조."""
    flux = result['flux_norm']
    pred = result['pred_ens']
    ax.plot(TARGET_WAVE, flux, color='#1f77b4', lw=0.7, alpha=0.9,
            zorder=3)
    ax.axhline(1.0, color='#999', ls='--', lw=0.6, alpha=0.5)
    ymin = max(0.0, float(flux.min()) - 0.05)
    ymax = min(5.0, float(flux.max()) + 0.28)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(WAVE_MIN, WAVE_MAX)
    ax.set_xlabel("파장 (Å)")
    ax.set_ylabel("정규화 flux")

    if show_lines:
        yr = ymax - ymin
        for k, (name, center, lo, hi, species, ko, sens) in \
                enumerate(LINES_V5):
            is_key = highlight and pred in LINE_SENS[name] \
                and result['ews'].get(name, 0) > 0.15
            color = '#c0392b' if is_key else '#999'
            lw_ = 1.1 if is_key else 0.4
            alpha = 0.9 if is_key else 0.35
            ax.axvline(center, color=color, ls='--', lw=lw_, alpha=alpha,
                       zorder=2)
            # 라벨 높이를 2단으로 번갈아 배치 (겹침 방지)
            ytxt = ymax - yr * (0.02 if k % 2 == 0 else 0.11)
            ax.text(center, ytxt, ko, rotation=90, fontsize=6.5,
                    color='#8e1e12' if is_key else '#777',
                    fontweight='bold' if is_key else 'normal',
                    ha='right', va='top', alpha=0.95 if is_key else 0.6,
                    zorder=4)
    title = f"{sid}"
    if highlight:
        title += (f"  —  {pred}형 판정 근거 선은 붉은색")
    ax.set_title(title, fontsize=10, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════
def run_gui(open_path=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:
        plt.rc('font', family='Malgun Gothic')
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

    class App:
        def __init__(self, root):
            self.root = root
            root.title("항성 분광형 분류 v5 — OB/A/F/G/K/M + 광도계급 + 원소선")
            root.geometry("1320x820")

            self.pred = None
            self.src = None
            self.idx = 0
            self.result = None
            self.cur = None

            self._build_ui()
            self._load_models()
            if open_path:
                root.after(200, lambda: self._open(open_path))

        def _build_ui(self):
            top = tk.Frame(self.root, padx=8, pady=6)
            top.pack(side="top", fill="x")

            ttk.Button(top, text="📂 파일 열기...",
                       command=self.on_open).pack(side="left")
            ttk.Button(top, text="📁 폴더 열기...",
                       command=self.on_open_dir).pack(side="left", padx=(4, 0))
            self.prev_btn = ttk.Button(top, text="◀", width=3,
                                       command=self.on_prev, state="disabled")
            self.prev_btn.pack(side="left", padx=(8, 0))
            self.next_btn = ttk.Button(top, text="▶", width=3,
                                       command=self.on_next, state="disabled")
            self.next_btn.pack(side="left", padx=(2, 8))
            self.pos_label = ttk.Label(top, text="- / -")
            self.pos_label.pack(side="left")

            self.spec_var = tk.StringVar(value="—")
            self.spec_combo = ttk.Combobox(top, textvariable=self.spec_var,
                                           width=24, state="disabled")
            self.spec_combo.pack(side="left", padx=(8, 2))
            self.spec_combo.bind("<<ComboboxSelected>>", self.on_combo_select)

            self.search_var = tk.StringVar()
            se = ttk.Entry(top, textvariable=self.search_var, width=11)
            se.pack(side="left", padx=(6, 2))
            se.bind("<Return>", self.on_search)
            self.search_btn = ttk.Button(top, text="🔎", width=3,
                                         command=self.on_search,
                                         state="disabled")
            self.search_btn.pack(side="left")

            def entry(parent, label, width=7):
                ttk.Label(parent, text=label).pack(side="left", padx=(10, 2))
                v = tk.StringVar()
                ttk.Entry(parent, textvariable=v, width=width).pack(side="left")
                return v
            self.rv_var = entry(top, "RV(km/s):")
            self.av_var = entry(top, "A_V:", 5)
            self.ra_var = entry(top, "적경(°):", 8)
            self.dec_var = entry(top, "적위(°):", 8)
            ttk.Button(top, text="🌐 SIMBAD 대조",
                       command=self.on_simbad).pack(side="left", padx=(6, 0))
            ttk.Button(top, text="🔄 재분류",
                       command=self.reclassify).pack(side="left", padx=8)
            self.batch_btn = ttk.Button(top, text="📊 전체 분류 → CSV",
                                        command=self.on_batch,
                                        state="disabled")
            self.batch_btn.pack(side="left", padx=(4, 0))
            ttk.Button(top, text="🖼 그림 저장",
                       command=self.on_savefig).pack(side="left", padx=(4, 0))

            self.status = ttk.Label(top, text="모델 로드 중...",
                                    foreground="#666")
            self.status.pack(side="right")

            self.root.bind("<Left>", lambda e: self.on_prev())
            self.root.bind("<Right>", lambda e: self.on_next())

            # ── 판정 배너: 분광형 + 광도계급을 크게 강조 표시 ──
            self.banner = tk.Frame(self.root, bg="#f0f0f0", pady=6)
            self.banner.pack(side="top", fill="x", padx=8)
            self.big_label = tk.Label(
                self.banner, text="파일을 열면 판정 결과가 여기 표시됩니다",
                font=("Malgun Gothic", 24, "bold"),
                bg="#f0f0f0", fg="#666")
            self.big_label.pack()
            self.sub_label = tk.Label(
                self.banner, text="", font=("Malgun Gothic", 12),
                bg="#f0f0f0", fg="#444")
            self.sub_label.pack()

            main = ttk.PanedWindow(self.root, orient="horizontal")
            main.pack(side="top", fill="both", expand=True, padx=8, pady=4)

            left = ttk.LabelFrame(main, text="스펙트럼 + 원소 흡수선", padding=4)
            bar = tk.Frame(left)
            bar.pack(side="top", fill="x")
            self.view_var = tk.StringVar(value="norm")
            ttk.Radiobutton(bar, text="정규화 (모델 입력)", value="norm",
                            variable=self.view_var,
                            command=self.redraw_spec).pack(side="left")
            ttk.Radiobutton(bar, text="원본 (raw flux)", value="raw",
                            variable=self.view_var,
                            command=self.redraw_spec).pack(side="left")
            self.lines_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="원소선 표시", variable=self.lines_var,
                            command=self.redraw_spec).pack(side="left",
                                                           padx=(12, 0))
            self.hl_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="판정 근거 강조", variable=self.hl_var,
                            command=self.redraw_spec).pack(side="left",
                                                           padx=(6, 0))
            self.fig_spec = Figure(figsize=(7.8, 5.5), dpi=100)
            self.ax_spec = self.fig_spec.add_subplot(111)
            self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=left)
            self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)
            main.add(left, weight=3)

            right = ttk.LabelFrame(main, text="분광형 확률 + 물리량", padding=4)
            self.fig_prob = Figure(figsize=(5.2, 5.5), dpi=100)
            self.ax_prob = self.fig_prob.add_subplot(111)
            self.canvas_prob = FigureCanvasTkAgg(self.fig_prob, master=right)
            self.canvas_prob.get_tk_widget().pack(fill="both", expand=True)
            main.add(right, weight=2)

            bot = ttk.Frame(self.root, padding=(10, 0))
            bot.pack(side="bottom", fill="x")
            self.result_text = tk.Text(bot, height=7, font=("Consolas", 11),
                                       background="#fafafa", relief="solid",
                                       borderwidth=1, padx=12, pady=8)
            self.result_text.pack(side="left", fill="x", expand=True,
                                  pady=(0, 8))
            self.result_text.insert("1.0", "파일을 열면 자동으로 분류합니다.\n")
            self.result_text.config(state="disabled")

        def _load_models(self):
            try:
                self.pred = PredictorV5()
                dev = "GPU" if DEVICE.type == "cuda" else "CPU"
                self.status.config(
                    text=f"✓ v5 모델 로드 완료 [{dev}] "
                         f"(6클래스 · {N_FEATURES}피처 · Teff+logg 예측)",
                    foreground="#070")
            except Exception as e:
                self.status.config(text=f"✗ 모델 로드 실패: {e}",
                                   foreground="#c00")
                from tkinter import messagebox
                messagebox.showerror("모델 로드 실패", str(e))

        def on_open(self):
            path = filedialog.askopenfilename(
                title="스펙트럼 파일 선택",
                filetypes=[("지원 파일", "*.fits *.fit *.csv *.txt *.dat *.npy"),
                           ("FITS", "*.fits *.fit"),
                           ("텍스트", "*.csv *.txt *.dat"),
                           ("NumPy", "*.npy"),
                           ("모든 파일", "*.*")])
            if path:
                self._open(path)

        def on_open_dir(self):
            path = filedialog.askdirectory(title="스펙트럼 폴더 선택")
            if path:
                self._open(path)

        def _open(self, path):
            if self.src is not None:
                self.src.close()
                self.src = None
            try:
                self.src = SpectrumSource(path)
            except Exception as e:
                messagebox.showerror("파일 열기 실패", str(e))
                return
            self.idx = 0
            self.batch_btn.config(
                state="normal" if self.src.n >= 1 else "disabled")
            n = self.src.n
            if n > 1:
                labels = [f"{i+1}/{n}  {sid}"
                          for i, sid in enumerate(self.src.ids[:5000])]
                if n > 5000:
                    labels.append(f"... ({n-5000:,}개 더 — 🔎 검색 사용)")
                self.spec_combo["values"] = labels
                self.spec_combo.config(state="readonly")
                self.spec_var.set(labels[0])
                self.search_btn.config(state="normal")
            else:
                self.spec_combo["values"] = ["—"]
                self.spec_combo.config(state="disabled")
                self.spec_var.set("—")
                self.search_btn.config(state="disabled")
            self._nav_update()
            self.classify_current()

        def _nav_update(self):
            n = self.src.n if self.src else 0
            self.pos_label.config(
                text=f"{self.idx + 1} / {n}  [{self.src.kind}]"
                if n else "- / -")
            if self.src and n > 1 and self.idx < 5000:
                self.spec_var.set(
                    f"{self.idx+1}/{n}  {self.src.ids[self.idx]}")
            self.prev_btn.config(
                state="normal" if self.src and self.idx > 0 else "disabled")
            self.next_btn.config(
                state="normal" if self.src and self.idx < n - 1
                else "disabled")

        def on_combo_select(self, event=None):
            sel = self.spec_combo.current()
            if self.src and 0 <= sel < min(self.src.n, 5000):
                self.idx = sel
                self._nav_update()
                self.classify_current()

        def on_search(self, event=None):
            if not self.src:
                return
            q = self.search_var.get().strip().lower()
            if not q:
                return
            ids = self.src.ids
            hit = next((i for i, s in enumerate(ids)
                        if s.lower() == q), None)
            if hit is None:
                hit = next((i for i, s in enumerate(ids)
                            if q in s.lower()), None)
            if hit is None:
                messagebox.showinfo("검색",
                                    f"'{q}' 와 일치하는 별 ID가 없습니다.")
                return
            self.idx = hit
            self._nav_update()
            self.classify_current()

        def on_prev(self):
            if self.src and self.idx > 0:
                self.idx -= 1
                self._nav_update()
                self.classify_current()

        def on_next(self):
            if self.src and self.idx < self.src.n - 1:
                self.idx += 1
                self._nav_update()
                self.classify_current()

        def _float_or_none(self, var):
            s = var.get().strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None

        def classify_current(self):
            if self.pred is None or self.src is None:
                return
            try:
                self.cur = self.src.get(self.idx)
            except Exception as e:
                messagebox.showerror("읽기 실패", str(e))
                return
            sid, wave, flux, rv_file, av_file = self.cur
            rv = self._float_or_none(self.rv_var)
            av = self._float_or_none(self.av_var)
            rv = rv if rv is not None else rv_file
            av = av if av is not None else av_file

            # 좌표 자동 채움 (파일 헤더에 있으면 항상 갱신)
            ra, dec = self.src.get_radec(self.idx)
            if ra is not None:
                self.ra_var.set(f"{ra:.5f}")
                self.dec_var.set(f"{dec:.5f}")

            # rv/av 가 없으면 스펙트럼에서 자동 추정 (흡수선 RV, 색지수 A_V)
            r = self.pred.predict_auto(wave, flux, rv, av)
            if r is None:
                messagebox.showwarning(
                    "전처리 실패",
                    "유효 파장 구간(4000~7400Å) 데이터가 부족하거나 "
                    "플럭스가 비정상입니다.")
                return
            self.result = r
            self.redraw_spec()
            self.redraw_prob()
            self.update_text(sid, wave)

        def reclassify(self):
            self.classify_current()

        def on_simbad(self):
            """입력/자동 좌표로 SIMBAD 등록 정보를 조회해 AI 판정과 대조."""
            ra = self._float_or_none(self.ra_var)
            dec = self._float_or_none(self.dec_var)
            if ra is None or dec is None:
                messagebox.showinfo(
                    "SIMBAD 대조",
                    "적경/적위(도 단위)가 필요합니다.\n"
                    "서베이 fits 는 파일에서 자동으로 채워지고,\n"
                    "그 외에는 직접 입력하세요.")
                return
            info = simbad_lookup(ra, dec)
            if info is None:
                messagebox.showerror("SIMBAD 대조",
                                     "SIMBAD 서버 연결 실패 (인터넷 확인)")
                return
            if info.get('main_id') is None:
                messagebox.showinfo("SIMBAD 대조",
                                    "반경 3″ 안에 등록된 천체가 없습니다.")
                return
            sp = info.get('sp_type', '') or '(미등록)'
            msg = [f"SIMBAD 등록명: {info['main_id']}"
                   f"  (이격 {info['sep_arcsec']:.2f}″)",
                   f"등록 분광형: {sp}   천체형: {info.get('otype', '')}"]
            if self.result is not None:
                r = self.result
                msg.append(f"AI 판정   : {r['pred_ens']}형 "
                           f"(약 {r['teff_ens']:,.0f} K) · "
                           f"{LUM_KO.get(r['lum'], r['lum'])}")
                letter = re.search(r'[OBAFGKM]', sp or '')
                if letter:
                    sim_c = ('OB' if letter.group(0) in 'OB'
                             else letter.group(0))
                    ok = sim_c == r['pred_ens']
                    msg.append("→ " + ("일치 ✓" if ok else
                               "불일치 — SIMBAD 라벨 재검토 후보일 수 있음 "
                               "(문헌 Teff 확인 권장)"))
            messagebox.showinfo("SIMBAD 대조", "\n".join(msg))

        def redraw_spec(self):
            if self.result is None or self.cur is None:
                return
            sid, wave, flux, _, _ = self.cur
            ax = self.ax_spec
            ax.clear()
            if self.view_var.get() == "norm":
                draw_spectrum(ax, self.result, sid,
                              show_lines=self.lines_var.get(),
                              highlight=self.hl_var.get())
            else:
                m = np.isfinite(wave) & np.isfinite(flux)
                ax.plot(wave[m], flux[m], color='#555', lw=0.5)
                ax.set_ylabel("원본 flux")
                ax.set_xlabel("파장 (Å)")
                ax.set_title(f"{sid}", fontsize=10, fontweight='bold')
            self.fig_spec.tight_layout()
            self.canvas_spec.draw()

        def redraw_prob(self):
            r = self.result
            ax = self.ax_prob
            ax.clear()
            x = np.arange(N_CLASSES)
            w = 0.27
            ax.bar(x - w, r['probs_cnn'], w, label='CNN(ResNet)',
                   color='#8fb8de', alpha=0.9)
            ax.bar(x, r['probs_mlp'], w, label='MLP(흡수선)',
                   color='#f0b27a', alpha=0.9)
            ax.bar(x + w, r['probs_ens'], w, label='★ 앙상블',
                   color='#c0392b', alpha=0.9)
            k = int(r['probs_ens'].argmax())
            ax.bar([k + w], [r['probs_ens'][k]], w,
                   color='#8e1e12', edgecolor='k', lw=1.2)
            ax.text(k + w, r['probs_ens'][k] + 0.02,
                    f"{r['probs_ens'][k]*100:.0f}%",
                    ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_ORDER)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("확률")
            ax.legend(fontsize=8, loc='upper right')
            lum_ko = LUM_KO.get(r['lum'], r['lum'])
            ax.set_title(
                f"{r['pred_ens']}형 (약 {r['teff_ens']:,.0f} K) · {lum_ko}",
                fontsize=12, fontweight='bold')
            self.fig_prob.tight_layout()
            self.canvas_prob.draw()

        def update_text(self, sid, wave):
            r = self.result
            # ── 판정 배너 갱신 (크게 강조) ──
            lum_big = LUM_KO.get(r['lum'], r['lum'])
            bg = CLASS_BG.get(r['pred_ens'], '#f0f0f0')
            self.banner.config(bg=bg)
            self.big_label.config(
                text=f"{r['pred_ens']}형  ·  {lum_big}",
                bg=bg, fg="#1a1a2e")
            o_note = "  (O급 가능성)" if r['o_hint'] else ""
            self.sub_label.config(
                text=f"약 {r['teff_ens']:,.0f} K{o_note}   |   "
                     f"신뢰도 {r['prob_ens']*100:.0f}%   |   "
                     f"logg {r['logg']:.2f}",
                bg=bg, fg="#333")
            rv = r.get('rv_used')
            av = r.get('av_used')
            rv_tag = " (흡수선 자동추정)" if r.get('rv_auto') else ""
            av_tag = " (색지수 자동추정)" if r.get('av_auto') else ""
            lum_ko = LUM_KO.get(r['lum'], r['lum'])
            lines = [
                f"파일   : {self.src.name}  [{self.src.kind}]   ID: {sid}",
                f"입력   : {np.nanmin(wave):.0f}~{np.nanmax(wave):.0f}Å"
                f"  |  RV: "
                f"{f'{rv:.1f} km/s{rv_tag}' if rv is not None else '없음'}"
                f"  |  A_V: "
                f"{f'{av:.2f}{av_tag}' if av is not None else '~0 (보정 불필요)'}",
                f"★ 판정 : {r['pred_ens']}형 (약 {r['teff_ens']:,.0f} K) · "
                f"{lum_ko}   신뢰도 {r['prob_ens']*100:.1f}%",
                f"  CNN {r['pred_cnn']} ({r['prob_cnn']*100:.0f}%) · "
                f"MLP {r['pred_mlp']} ({r['prob_mlp']*100:.0f}%)"
                f" · logg 예측 {r['logg']:.2f} (선폭에서 추정)"
                f" · 광도계급 신뢰도 {r['lum_prob']*100:.0f}%",
            ]
            # 판정 근거 원소선 (EW 상위)
            key = [(name, r['ews'][name]) for name in r['ews']
                   if r['pred_ens'] in LINE_SENS[name]
                   and r['ews'][name] > 0.15]
            key.sort(key=lambda t: -t[1])
            ko_map = {l[0]: l[5] for l in LINES_V5}
            if key:
                s = ", ".join(f"{ko_map[n]} (EW {v:.1f}Å)"
                              for n, v in key[:5])
                lines.append(f"근거 선: {s}")
            if r['o_hint']:
                lines.append("· He II 흡수선 검출 + 고온 → OB 중에서도 "
                             "O급 가능성")
            if r['lum'] == 'wd':
                lines.append("⚠ 백색왜성 판정 — 학습 표본이 288개뿐이라 "
                             "신뢰도가 제한적임 (참고용)")
            # 특이 천체 경고: 두 모델의 판정 불일치 또는 낮은 앙상블 신뢰도
            # (실측: DA 백색왜성 PM J19141+4936 에서 CNN F/MLP OB 불일치
            #  + 54% 신뢰도 — 이런 별은 정상 분류를 신뢰하면 안 됨)
            if r['pred_cnn'] != r['pred_mlp'] or r['prob_ens'] < 0.65:
                lines.append("⚠ CNN/MLP 판정 불일치 또는 신뢰도 65% 미만 — "
                             "특이 천체(백색왜성·탄소별·저금속성 등) 가능성. "
                             "🌐 SIMBAD 대조로 등록 정보 확인 권장")
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "\n".join(lines))
            self.result_text.config(state="disabled")

        def on_savefig(self):
            """현재 스펙트럼 + 원소선 라벨 그림 PNG 저장 (보고서 그림용)."""
            if self.result is None:
                return
            sid = self.cur[0]
            default = f"spectrum_{re.sub(r'[^A-Za-z0-9_-]', '_', str(sid))}.png"
            out = filedialog.asksaveasfilename(
                title="그림 저장", defaultextension=".png",
                initialfile=default,
                filetypes=[("PNG", "*.png")])
            if not out:
                return
            import matplotlib.pyplot as plt2
            fig, ax = plt2.subplots(figsize=(12, 5), dpi=150)
            draw_spectrum(ax, self.result, sid, show_lines=True,
                          highlight=True)
            r = self.result
            lum_ko = LUM_KO.get(r['lum'], r['lum'])
            ax.set_title(f"{sid} — {r['pred_ens']}형 "
                         f"(약 {r['teff_ens']:,.0f} K) · {lum_ko}",
                         fontsize=12, fontweight='bold')
            fig.tight_layout()
            fig.savefig(out, bbox_inches='tight')
            plt2.close(fig)
            messagebox.showinfo("저장 완료", out)

        def on_batch(self):
            if self.src is None or self.pred is None:
                return
            default = os.path.splitext(self.src.name)[0] + \
                "_v5_predictions.csv"
            out = filedialog.asksaveasfilename(
                title="결과 CSV 저장", defaultextension=".csv",
                initialfile=default,
                filetypes=[("CSV", "*.csv"), ("모든 파일", "*.*")])
            if not out:
                return
            win = tk.Toplevel(self.root)
            win.title("일괄 분류 진행 중...")
            win.geometry("420x110")
            win.transient(self.root)
            pbar = ttk.Progressbar(win, length=380, mode="determinate",
                                   maximum=self.src.n)
            pbar.pack(pady=(18, 6))
            lab = ttk.Label(win, text="시작...")
            lab.pack()
            win.update()

            def cb(i, n):
                if i % 20 == 0 or i == n:
                    pbar['value'] = i
                    lab.config(text=f"{i:,} / {n:,}")
                    win.update()

            try:
                n = batch_classify(self.pred, self.src, out, progress=cb)
                win.destroy()
                messagebox.showinfo("완료", f"{n:,}개 분류 완료\n저장: {out}")
            except Exception as e:
                win.destroy()
                messagebox.showerror("일괄 분류 실패", str(e))

    root = tk.Tk()
    App(root)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="열 스펙트럼 파일/폴더")
    ap.add_argument("--batch", metavar="OUT_CSV",
                    help="GUI 없이 일괄 분류 후 CSV 저장")
    ap.add_argument("--models", help="models 폴더 경로 직접 지정")
    args = ap.parse_args()

    if args.batch:
        if not args.path:
            print("일괄 분류에는 입력 파일이 필요합니다.")
            sys.exit(1)
        pred = PredictorV5(args.models)
        src = SpectrumSource(args.path)
        print(f"{src.name} [{src.kind}] {src.n:,}개 분류 중...")
        n = batch_classify(pred, src, args.batch,
                           progress=lambda i, t: (i % 500 == 0 or i == t) and
                           print(f"  {i:,}/{t:,}", flush=True))
        print(f"완료: {n:,}행 → {args.batch}")
        return

    run_gui(args.path)


if __name__ == "__main__":
    main()
