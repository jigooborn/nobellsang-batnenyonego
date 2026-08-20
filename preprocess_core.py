# -*- coding: utf-8 -*-
"""
preprocess_core.py — v5 공통 전처리·피처 모듈 (단일 소스)

★ 이 파일이 v5의 유일한 전처리 구현이다.
  step1_v5.py(학습 데이터 생성)와 classify_gui_v5.py(추론 GUI)가
  둘 다 여기서 import 한다. 복사-수정 금지 (v4 함정 3번:
  SG 윈도우 하나 달랐던 것만으로 학습-추론 성능이 어긋났음).

v4 대비 변경 (2026-08-04, v5 설계):
  1) 성간 소광 보정 복원 — A_V 를 인자로 받아 Cardelli(1989, R_V=3.1)
     역적용. v3 에서 뺐던 이유(서베이 간 비일관)는 Gaia DR3 ag_gspphot
     으로 전 서베이 A_V 를 확보해 해소. av=None 이면 미보정(A/B 비교용).
  2) 흡수선 확장 — Liu et al. (2015) Table 2 의 측정 대역 26개
     + He II 4686(O급 감별) = 27개. 깊이 대신 등가폭(EW, Å) 측정.
  3) 선폭 피처 신규 — 발머 3선 + Mg b + Na D 의 FWHM·코어/날개 비율
     (압력 넓어짐 → 표면중력. 광도계급 분류의 이론적 근거를 직접 구현)
  4) 분류 경계 — 교과서(지구과학Ⅰ) 기준: OB>10000 / A>7500 / F>6000 /
     G>5000 / K>3500 / M. O·B 는 OB 한 클래스 (6클래스).
  5) 광도계급 — 초거성(logg<1.5) / 거성(<3.5) / 주계열(3.5~5.5).
     콤팩트(>5.5)는 학습에서 제외하되 함수는 값을 돌려줌(GUI 경고용).

전처리 순서 자체는 v3 검증본 그대로:
  배드픽셀 보간 → 도플러 → [소광 보정] → 4000~7400Å 컷 → κ-σ(+3σ만)
  → SG(win9, poly3) → 연속선 정규화(3차 다항 반복 피팅) → 붕괴 가드
  → 3401px 리샘플 → 피처 추출
"""

import numpy as np
from scipy.signal import savgol_filter

# ── 격자·전처리 상수 (v3 검증값 — 변경 금지) ───────────────────────────
C_KMS      = 299792.458
WAVE_MIN   = 4000.0
WAVE_MAX   = 7400.0
N_PIX      = 3401            # 1Å/픽셀
RV_EXT     = 3.1             # Cardelli R_V
SIGMA_CUT  = 3.0
KSIG_ITERS = 3
CONT_ITERS = 3
CONT_FLOOR = 0.05
SG_WIN     = 9               # Mg b 삼중선 병합 방지 (15는 병합 실측됨)

TARGET_WAVE = np.linspace(WAVE_MIN, WAVE_MAX, N_PIX)

# ── v5 분류 체계 ────────────────────────────────────────────────────────
# 교과서(분광형 분류 온도.png) 경계. O·B 통합 → 6클래스.
CLASS_ORDER = ['OB', 'A', 'F', 'G', 'K', 'M']
N_CLASSES   = len(CLASS_ORDER)
# 경계 (내림차순). 라벨 버퍼(prep)와 GUI 표시가 같이 씀.
BOUNDARIES  = [10000.0, 7500.0, 6000.0, 5000.0, 3500.0]
O_TEFF      = 30000.0        # OB 내부에서 "O급" 참고 표시용

LUM_ORDER = ['giant', 'ms', 'wd']    # 광도계급 3클래스 (사용자 결정 v5.1)
N_LUM     = len(LUM_ORDER)
LUM_KO    = {'giant': '거성', 'ms': '주계열',
             'wd': '백색왜성(콤팩트)', 'unknown': '불명'}
LOGG_GIANT   = 3.5
LOGG_COMPACT = 5.5           # 이 위는 백색왜성(콤팩트) — 표본 288개뿐, 성능 제한적


def teff_to_class(teff):
    """Teff → 6클래스 분광형 (교과서 경계)."""
    if teff > 10000: return 'OB'
    if teff >  7500: return 'A'
    if teff >  6000: return 'F'
    if teff >  5000: return 'G'
    if teff >  3500: return 'K'
    return 'M'


def logg_to_lum(g):
    """logg → 광도계급 라벨: 거성(<3.5) / 주계열(3.5~5.5) / 백색왜성(>5.5)."""
    if g is None or not np.isfinite(g):
        return 'unknown'
    if g > LOGG_COMPACT: return 'wd'
    if g < LOGG_GIANT:   return 'giant'
    return 'ms'


def near_boundary(teff, buffer_k=75.0):
    """경계 ±버퍼 안이면 True (라벨 노이즈 억제용 — prep 에서 제외)."""
    return any(abs(teff - b) <= buffer_k for b in BOUNDARIES)


# ── 흡수선 정의 ─────────────────────────────────────────────────────────
# Liu et al. (2015) Table 2 측정 대역(4000~7400Å 내 26개) + He II 4686.
# (name, 중심Å, 대역lo, 대역hi, 화학종, 한글표시, 민감 분광형)
# EW 는 정규화 스펙트럼에서 sum(1-flux)dλ 로 측정.
LINES_V5 = [
    ('Hdelta',   4101.7, 4083.50, 4122.25, 'H I',   'Hδ (수소)',        'OB~A'),
    ('CN4160',   4160.0, 4143.38, 4178.38, 'CN',    'CN 분자띠',        'G~K'),
    ('Ca4227',   4226.7, 4223.50, 4236.00, 'Ca I',  'Ca I 4227',        'F~M'),
    ('Gband',    4300.0, 4282.63, 4317.63, 'CH',    'G밴드 (CH)',       'F~K'),
    ('Hgamma',   4340.5, 4319.75, 4363.50, 'H I',   'Hγ (수소)',        'OB~A'),
    ('Fe4383',   4383.0, 4370.38, 4421.63, 'Fe I',  'Fe I 4383',        'F~K'),
    ('He4388',   4387.9, 4381.00, 4399.00, 'He I',  'He I 4388',        'OB'),
    ('Ca4455',   4455.0, 4453.38, 4475.88, 'Ca I',  'Ca I 4455',        'F~K'),
    ('He4471',   4471.5, 4462.00, 4475.00, 'He I',  'He I 4471',        'OB'),
    ('Fe4531',   4531.0, 4515.50, 4560.50, 'Fe I',  'Fe I 4531',        'F~K'),
    ('He4542',   4541.6, 4536.00, 4548.00, 'He II', 'He II 4542',       'O급'),
    ('Fe4668',   4668.0, 4635.25, 4721.50, 'Fe I',  'Fe I 4668 (+C₂)',  'G~K'),
    ('He4686',   4685.7, 4679.00, 4693.00, 'He II', 'He II 4686',       'O급'),
    ('Hbeta',    4861.3, 4847.88, 4876.63, 'H I',   'Hβ (수소)',        'OB~A'),
    ('Fe5015',   5015.0, 4977.75, 5054.00, 'Fe I',  'Fe I 5015',        'G~K'),
    ('Mg1',      5100.0, 5069.13, 5134.13, 'MgH',   'Mg₁ (MgH)',        'K~M'),
    ('Mg2',      5175.0, 5154.13, 5196.63, 'Mg+MgH','Mg₂',              'K~M'),
    ('Mgb',      5176.4, 5160.13, 5192.63, 'Mg I',  'Mg b 삼중선',      'G~M'),
    ('Fe5270',   5270.0, 5245.65, 5285.65, 'Fe I',  'Fe I 5270',        'G~K'),
    ('Fe5335',   5335.0, 5312.13, 5352.13, 'Fe I',  'Fe I 5335',        'G~K'),
    ('Fe5406',   5406.0, 5387.50, 5415.00, 'Fe I',  'Fe I 5406',        'G~K'),
    ('Fe5709',   5709.0, 5698.38, 5722.13, 'Fe I',  'Fe I 5709',        'G~K'),
    ('Fe5782',   5782.0, 5778.38, 5798.38, 'Fe I',  'Fe I 5782',        'G~K'),
    ('NaD',      5893.0, 5878.63, 5911.13, 'Na I',  'Na D 이중선',      'K~M'),
    ('TiO1',     5965.0, 5938.38, 5995.88, 'TiO',   'TiO₁ 분자띠',      'M'),
    ('TiO2',     6230.0, 6191.38, 6273.88, 'TiO',   'TiO₂ 분자띠',      'M'),
    ('Halpha',   6562.8, 6548.00, 6578.00, 'H I',   'Hα (수소)',        'OB~A'),
]
LINE_NAMES = [l[0] for l in LINES_V5]

# 선폭(FWHM·코어/날개) 측정 대상 — 압력 넓어짐이 큰 선들
#   발머선: 고온별 광도계급 (린어 스타르크 넓어짐)
#   Mg b / Na D: 저온별 광도계급 (반데르발스 넓어짐; 왜성일수록 날개 발달)
WIDTH_LINES = [
    ('Hdelta', 4101.7), ('Hgamma', 4340.5), ('Hbeta', 4861.3),
    ('Mgb', 5176.4), ('NaD', 5893.0),
]

# 색지수 대역 (빈의 변위 법칙 — 정규화 전 연속선에서 측정)
COLOR_BLUE  = (4000.0, 4600.0)
COLOR_GREEN = (5300.0, 5900.0)
COLOR_RED   = (6800.0, 7400.0)

FEATURE_NAMES = (
    [f'ew_{n}' for n in LINE_NAMES] +
    [f'fwhm_{n}' for n, _ in WIDTH_LINES] +
    [f'corewing_{n}' for n, _ in WIDTH_LINES] +
    ['cont_slope', 'cont_curve']
)
N_FEATURES = len(FEATURE_NAMES)   # 27 + 5 + 5 + 2 = 39


# ── 헬퍼 (v3 검증본 그대로) ─────────────────────────────────────────────

def safe_arr(arr):
    a = np.asarray(arr)
    if a.dtype.byteorder not in ('=', '|'):
        a = a.byteswap().view(a.dtype.newbyteorder())
    return np.ascontiguousarray(a, dtype=np.float64)


def vac_to_air(wave_vac):
    """진공→공기 파장 (Morton 1991 IAU 근사식)."""
    s2 = (1e4 / wave_vac) ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return wave_vac / n


def cardelli_alav(wave_ang):
    """Cardelli+(1989) 소광 곡선 A(λ)/A_V (R_V=3.1)."""
    wave_um = np.asarray(wave_ang, dtype=np.float64) / 1e4
    x = 1.0 / np.where(wave_um > 0, wave_um, 1e-6)
    a = np.zeros_like(x); b = np.zeros_like(x)
    m = (x >= 0.3) & (x <= 1.1)
    a[m] = 0.574 * x[m] ** 1.61
    b[m] = -0.527 * x[m] ** 1.61
    m = (x > 1.1) & (x <= 3.3)
    y = x[m] - 1.82
    a[m] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
              + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[m] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    m = (x > 3.3) & (x <= 8.0)
    fa = np.zeros_like(x); fb = np.zeros_like(x)
    m2 = (x > 5.9) & (x <= 8.0)
    y2 = x[m2] - 5.9
    fa[m2] = -0.04473 * y2**2 - 0.009779 * y2**3
    fb[m2] =  0.2130  * y2**2 + 0.1207   * y2**3
    a[m] = (1.752 - 0.316*x[m] - 0.104 / ((x[m] - 4.67)**2 + 0.341) + fa[m])
    b[m] = (-3.090 + 1.825*x[m] + 1.206 / ((x[m] - 4.62)**2 + 0.263) + fb[m])
    m = (x > 8.0) & (x <= 10.0)
    y3 = x[m] - 8.0
    a[m] = -1.073 - 0.628*y3 + 0.137*y3**2 - 0.070*y3**3
    b[m] =  13.670 + 4.257*y3 - 0.420*y3**2 + 0.374*y3**3
    return a + b / RV_EXT


def fit_continuum(wave_norm, flux, deg=3, iters=CONT_ITERS, kappa=2.5):
    mask = np.isfinite(flux) & (flux > 0)
    if mask.sum() < deg + 5:
        return np.ones_like(flux)
    for _ in range(iters):
        try:
            coeffs = np.polyfit(wave_norm[mask], flux[mask], deg=deg)
        except (np.linalg.LinAlgError, ValueError):
            return np.ones_like(flux)
        cont = np.polyval(coeffs, wave_norm)
        resid = flux - cont
        s = np.std(resid[mask])
        if s == 0 or not np.isfinite(s):
            break
        new_mask = mask & (resid > -kappa * s) & (resid < 2.0 * s)
        if new_mask.sum() < deg + 5 or new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    coeffs = np.polyfit(wave_norm[mask], flux[mask], deg=deg)
    cont = np.polyval(coeffs, wave_norm)
    # 0-나눗셈 가드 — v5: 절대값 1e-10 대신 스케일 상대 기준.
    # (절대 기준은 플럭스가 erg 단위(~1e-15)로 들어오면 연속선을 통째로
    #  1e-10 으로 치환해 정규화를 붕괴시킴 — GUI 임의 단위 입력 대비)
    floor = 1e-10 * max(float(np.nanmedian(np.abs(cont))), 1e-300)
    return np.where(np.abs(cont) < floor, np.sign(cont) * floor + (cont == 0) * floor, cont)


def kappa_sigma_clip(flux, cont, kappa=SIGMA_CUT, iters=KSIG_ITERS):
    out = flux.copy()
    norm = out / cont
    for _ in range(iters):
        resid = norm - 1.0
        s = np.std(resid)
        if s == 0 or not np.isfinite(s):
            break
        bad = resid > kappa * s
        if not bad.any():
            break
        out[bad] = cont[bad]
        norm = out / cont
    return out


# ── 피처 측정 ───────────────────────────────────────────────────────────

def _ew(flux_norm, lo, hi):
    """등가폭(Å): 정규화 스펙트럼에서 sum(1-flux)dλ. 1Å/px 격자."""
    i0 = int(np.searchsorted(TARGET_WAVE, lo))
    i1 = int(np.searchsorted(TARGET_WAVE, hi))
    i0 = int(np.clip(i0, 0, N_PIX - 2))
    i1 = int(np.clip(i1, i0 + 1, N_PIX))
    return float(np.sum(1.0 - flux_norm[i0:i1]))   # dλ=1Å


def _line_width(flux_norm, center, search=8.0, wing=(5.0, 12.0)):
    """선폭 2종: (FWHM Å, 코어/날개 깊이 비).
    - 중심 ±search Å 안의 최소점을 코어로 잡음
    - FWHM: 반깊이 교차점을 좌우 탐색 (교차 못 찾으면 탐색폭 상한)
    - 코어/날개: 코어 ±2Å 평균 깊이 vs 날개(5~12Å) 평균 깊이의 로그비
      (왜성 = 날개 발달 → 비율 작음, 초거성 = 좁고 깊음 → 비율 큼)"""
    ic = int(round(center - WAVE_MIN))
    if ic < 15 or ic > N_PIX - 16:
        return 0.0, 0.0
    s = int(search)
    seg = flux_norm[ic - s: ic + s + 1]
    imin = int(np.argmin(seg))
    ic2 = ic - s + imin                    # 실제 코어 위치
    depth = 1.0 - float(flux_norm[ic2])
    if depth < 0.01:                       # 선이 사실상 없음
        return 0.0, 0.0
    half = 1.0 - depth / 2.0
    # 좌우로 반깊이 교차 탐색 (최대 25Å)
    max_r = 25
    left = right = max_r
    for k in range(1, max_r + 1):
        if ic2 - k < 0 or flux_norm[ic2 - k] >= half:
            left = k; break
    for k in range(1, max_r + 1):
        if ic2 + k >= N_PIX or flux_norm[ic2 + k] >= half:
            right = k; break
    fwhm = float(left + right)             # 1Å/px
    # 코어/날개 깊이
    core = 1.0 - float(np.mean(flux_norm[ic2 - 2: ic2 + 3]))
    wl, wh = int(wing[0]), int(wing[1])
    wing_d = 1.0 - float(np.mean(np.concatenate([
        flux_norm[ic2 - wh: ic2 - wl + 1], flux_norm[ic2 + wl: ic2 + wh + 1]])))
    cw = np.log10(max(core, 1e-3) / max(wing_d, 1e-3))
    return float(np.clip(fwhm, 0.0, 50.0)), float(np.clip(cw, -2.0, 2.0))


# ── 시선속도 자동 추정 (흡수선 기반) ────────────────────────────────────
# 헤더에 RV 가 없는 파일(일반 FITS/CSV)용. 강한 흡수선 7개의 코어 위치를
# 정지 파장과 비교해 도플러 이동을 역산한다. 포물선 보간으로 서브픽셀
# 정밀도 확보, 선별 조건(깊이>3σ)과 중앙값+MAD 로 이상선 배제.
RV_LINES = [4101.7, 4226.7, 4340.5, 4861.3, 5183.6, 5892.9, 6562.8]


def estimate_rv(raw_wave, raw_flux, search_kms=600.0):
    """흡수선 코어 위치로 시선속도(km/s) 추정. 실패 시 None.
    반환값 부호: 양수 = 후퇴(적색이동)."""
    wave = np.asarray(raw_wave, dtype=np.float64)
    flux = np.asarray(raw_flux, dtype=np.float64)
    m = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    if m.sum() < 500:
        return None
    wave, flux = wave[m], flux[m]
    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]

    vs = []
    for lam0 in RV_LINES:
        half = lam0 * search_kms / C_KMS + 2.0
        sel = (wave >= lam0 - half) & (wave <= lam0 + half)
        if sel.sum() < 9:
            continue
        ww, ff = wave[sel], flux[sel]
        n_edge = max(3, len(ff) // 8)
        c0, c1 = np.median(ff[:n_edge]), np.median(ff[-n_edge:])
        if c0 <= 0 or c1 <= 0:
            continue
        cont = np.interp(ww, [ww[0], ww[-1]], [c0, c1])
        nf = ff / np.maximum(cont, 1e-300)
        i = int(np.argmin(nf))
        if i <= 0 or i >= len(nf) - 1:
            continue
        depth = 1.0 - nf[i]
        noise = float(np.std(np.diff(nf)) / np.sqrt(2.0))
        if depth < max(0.03, 3.0 * noise):
            continue
        y0, y1, y2 = nf[i - 1], nf[i], nf[i + 1]
        denom = y0 - 2.0 * y1 + y2
        d = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        d = float(np.clip(d, -1.0, 1.0))
        step = (ww[i + 1] - ww[i - 1]) / 2.0
        lam_min = ww[i] + d * step
        v = (lam_min / lam0 - 1.0) * C_KMS
        if abs(v) < search_kms:
            vs.append(v)
    if len(vs) < 3:
        return None
    vs = np.array(vs)
    med = float(np.median(vs))
    mad = float(np.median(np.abs(vs - med)))
    keep = vs[np.abs(vs - med) < max(3.0 * 1.4826 * mad, 30.0)]
    if len(keep) < 3:
        return None
    return float(np.median(keep))


# ── A_V 자동 추정 (색지수 초과분 역산) ──────────────────────────────────
# 원리: 흡수선이 주도하는 Teff 추정치 → 그 온도의 별이 가져야 할 연속선
# 기울기(무소광 학습 표본에서 보정한 곡선) → 실제 기울기와의 차이를
# Cardelli 소광 기울기 계수로 나눠 A_V 역산. ±0.2~0.3 등급 수준의 참고값.
_K_SLOPE = None


def slope_per_av():
    """A_V 1등급당 cont_slope(log10 적/청) 증가량 — Cardelli 에서 유도."""
    global _K_SLOPE
    if _K_SLOPE is None:
        b = float(np.mean(cardelli_alav(
            np.linspace(*COLOR_BLUE, 100))))
        r = float(np.mean(cardelli_alav(
            np.linspace(*COLOR_RED, 100))))
        _K_SLOPE = 0.4 * (b - r)     # ≈ 0.21 /mag
    return _K_SLOPE


def estimate_av(teff_pred, slope_obs, calib):
    """calib = (log10Teff 격자, 무소광 중앙 slope). 반환: A_V 추정 (0~3)."""
    if not (np.isfinite(teff_pred) and teff_pred > 1000):
        return None
    lt = np.log10(teff_pred)
    exp_slope = float(np.interp(lt, calib[0], calib[1]))
    av = (slope_obs - exp_slope) / slope_per_av()
    if not np.isfinite(av):
        return None
    return float(np.clip(av, 0.0, 3.0))


# ── 전처리 본체 ─────────────────────────────────────────────────────────

def process_one(raw_wave, raw_flux, rv_kms=None, av=None):
    """스펙트럼 1개 전처리 + 피처 추출.

    입력: 관측 파장(Å, 공기 기준으로 변환 완료 상태), 플럭스,
          시선속도(km/s, 없으면 None), A_V(mag, 없으면 None=미보정)
    출력: (flux_norm[3401] float32, features[N_FEATURES] float32) 또는 None
    """
    flux = np.asarray(raw_flux, dtype=np.float64).copy()
    wave = np.asarray(raw_wave, dtype=np.float64).copy()

    fw = np.isfinite(wave)
    if fw.mean() < 0.5 or fw.sum() < 200:
        return None
    wave, flux = wave[fw], flux[fw]

    # 배드픽셀(NaN/0/음수) → 이웃 선형 보간 (가짜 0-딥 방지)
    good = np.isfinite(flux) & (flux > 0)
    if good.mean() < 0.5:
        return None
    if not good.all():
        gi = np.where(good)[0]
        flux = np.interp(np.arange(len(flux)), gi, flux[gi])

    # 도플러 보정
    if rv_kms is not None and np.isfinite(rv_kms):
        denom = 1.0 + rv_kms / C_KMS
        if denom <= 0 or not np.isfinite(denom):
            return None
        wave = wave / denom

    # 성간 소광 보정 (v5: A_V 있으면 전 서베이 적용)
    if av is not None and np.isfinite(av) and av > 0.0:
        flux = flux * 10.0 ** (0.4 * av * cardelli_alav(wave))

    # 파장 컷
    mask = (wave >= WAVE_MIN) & (wave <= WAVE_MAX)
    if mask.sum() < 200:
        return None
    wave_c = wave[mask]
    flux_c = flux[mask]
    order = np.argsort(wave_c)
    wave_c, flux_c = wave_c[order], flux_c[order]

    wave_n = (wave_c - WAVE_MIN) / (WAVE_MAX - WAVE_MIN) * 2.0 - 1.0
    cont1 = fit_continuum(wave_n, flux_c, deg=3, iters=CONT_ITERS)
    flux_c = kappa_sigma_clip(flux_c, cont1, kappa=SIGMA_CUT, iters=KSIG_ITERS)

    win = min(SG_WIN, (len(flux_c) // 30) * 2 + 1)
    win = max(win, 5)
    if win % 2 == 0:
        win += 1
    flux_c = savgol_filter(flux_c, window_length=win, polyorder=3)

    cont2 = fit_continuum(wave_n, flux_c, deg=3, iters=CONT_ITERS)

    # 연속선 붕괴 가드
    med = np.median(cont2)
    if not np.isfinite(med) or med <= 0:
        return None
    lowc = cont2 < CONT_FLOOR * med
    flux_norm = flux_c / np.where(lowc, med, cont2)
    flux_norm[lowc] = 1.0
    flux_norm[~np.isfinite(flux_norm)] = 1.0

    flux_out = np.interp(TARGET_WAVE, wave_c, flux_norm, left=1.0, right=1.0)
    flux_out = np.clip(flux_out, 0.0, 5.0).astype(np.float32)

    # ── 피처 ──
    f64 = flux_out.astype(np.float64)
    ews = [_ew(f64, lo, hi) for _, _, lo, hi, _, _, _ in LINES_V5]

    fwhms, cws = [], []
    for _, center in WIDTH_LINES:
        fw_, cw_ = _line_width(f64, center)
        fwhms.append(fw_); cws.append(cw_)

    # 색지수 (정규화 전 연속선 cont2 — 소광 보정 후라 v5 에선 신뢰도↑)
    def band_mean(lo, hi):
        m = (wave_c >= lo) & (wave_c <= hi)
        if m.sum() < 10:
            return np.nan
        v = np.mean(cont2[m])
        return v if v > 0 else np.nan
    b = band_mean(*COLOR_BLUE)
    g = band_mean(*COLOR_GREEN)
    r = band_mean(*COLOR_RED)
    slope = np.log10(r / b) if (np.isfinite(b) and np.isfinite(r)) else 0.0
    curve = (np.log10(g * g / (b * r))
             if all(np.isfinite(x) for x in (b, g, r)) else 0.0)
    slope = float(np.clip(np.nan_to_num(slope), -2.0, 2.0))
    curve = float(np.clip(np.nan_to_num(curve), -2.0, 2.0))

    feats = np.array(ews + fwhms + cws + [slope, curve], dtype=np.float32)
    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats)
    return flux_out, feats


# ── 자가 테스트 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # 가짜 G형: 완만한 연속선 + 발머/Mgb/NaD 가우시안 흡수 + 노이즈
    wave = np.linspace(3900, 7500, 7200)
    cont = 1e-15 * (1.0 + 0.3 * (wave - 5000) / 2500)
    flux = cont.copy()
    for c, d, w in [(4101.7, .25, 3), (4340.5, .3, 3), (4861.3, .35, 3),
                    (6562.8, .4, 3), (5176.4, .3, 1.5), (5893.0, .25, 1.5),
                    (4300.0, .2, 5)]:
        flux *= 1 - d * np.exp(-0.5 * ((wave - c) / w) ** 2)
    flux *= 1 + rng.normal(0, 0.01, len(wave))
    flux[100:110] = 0.0          # 배드픽셀
    flux[3000] *= 30             # 우주선 스파이크

    res = process_one(wave, flux, rv_kms=30.0, av=0.5)
    assert res is not None
    fx, ft = res
    assert fx.shape == (N_PIX,) and ft.shape == (N_FEATURES,)
    d = dict(zip(FEATURE_NAMES, ft))
    print(f"OK — flux {fx.shape}, features {N_FEATURES}개")
    print(f"  EW: Hβ {d['ew_Hbeta']:.2f}Å  Mgb {d['ew_Mgb']:.2f}Å  "
          f"He4542 {d['ew_He4542']:.2f}Å (없어야 정상≈0)")
    print(f"  FWHM: Hβ {d['fwhm_Hbeta']:.1f}Å  코어/날개 {d['corewing_Hbeta']:.2f}")
    print(f"  색지수: slope {d['cont_slope']:.3f} curve {d['cont_curve']:.3f}")
    print(f"  스파이크 제거 확인: max={fx.max():.2f} (≤5)")
    assert d['ew_Hbeta'] > 1.0 and abs(d['ew_He4542']) < 0.5
    assert teff_to_class(5500) == 'G' and teff_to_class(12000) == 'OB'
    assert logg_to_lum(1.0) == 'supergiant' and logg_to_lum(4.4) == 'ms'
    print("자가 테스트 통과")
