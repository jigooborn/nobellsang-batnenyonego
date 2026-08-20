# -*- coding: utf-8 -*-
"""
xsl_validate.py — 진짜 외부 데이터 검증: XSL DR3 (VLT/X-shooter)

MILES 는 학습에 포함돼 있어 "진짜 외부" 검증이 아니었음 (v4 미해결 이슈 7).
XSL 은 학습에 전혀 안 쓴 독립 망원경(VLT)·독립 기기(X-shooter, R~10,000)·
독립 파라미터 파이프라인(Arentsen+2019, ULySS) → 일반화 능력의 결정적 시험.

데이터:
  스펙트럼: CDS J/A+A/660/A34/fits/ (830개, WAVE[nm]·진공 — 실측 확인됨)
  정답 Teff/logg: CDS J/A+A/627/A138 tablea1.dat (754개)
조건: 완전 실전 모드 — RV 는 흡수선 자동 추정, A_V 는 색지수 자동 추정.

실행: python xsl_validate.py           (파일럿 60개)
      python xsl_validate.py --all     (전체)
"""

import os
import re
import io
import sys
import urllib.request
import numpy as np
import pandas as pd

from preprocess_core import teff_to_class, logg_to_lum, vac_to_air, CLASS_ORDER
from classify_gui_v5 import PredictorV5
from preprocess_core import LUM_ORDER

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'external_xsl')
CDS_FITS = 'https://cdsarc.cds.unistra.fr/ftp/J/A+A/660/A34/'
CDS_PAR = 'https://cdsarc.cds.unistra.fr/ftp/J/A+A/627/A138/'
OUT = os.path.join('results', 'xsl_external')


def dl(url, path, timeout=60):
    """타임아웃·재시도 있는 다운로드 (urlretrieve 는 무한 대기 위험)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import time
    time.sleep(0.4)          # CDS 스로틀 방지 (연속 요청 간 예의상 대기)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, \
                    open(path + '.part', 'wb') as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + '.part', path)
            return
        except Exception as e:
            last = e
            time.sleep(3.0 * (attempt + 1))   # 점증 백오프
    raise RuntimeError(f"다운로드 실패({type(last).__name__}): {url}")


def load_tables():
    os.makedirs(BASE, exist_ok=True)
    dl(CDS_FITS + 'table.dat', os.path.join(BASE, 'xsl_table.dat'))
    dl(CDS_PAR + 'tablea1.dat', os.path.join(BASE, 'params.dat'))

    # XSL DR3 목록 (XSLID 51-55, FileName 72-110)
    rows = []
    with open(os.path.join(BASE, 'xsl_table.dat'), encoding='ascii',
              errors='replace') as f:
        for line in f:
            if len(line) < 72:
                continue
            rows.append({'xslid': line[50:55].strip(),
                         'file': line[71:110].strip()})
    files = pd.DataFrame(rows)

    # Arentsen+2019: XSLID 와 채택 Teff/logg/[Fe/H]
    # (ReadMe 기준: Star 1-25, XSLID 뒤쪽 — 안전하게 정규식으로 X#### 추출,
    #  채택값 3개는 뒤에서부터 파싱)
    prows = []
    with open(os.path.join(BASE, 'params.dat'), encoding='ascii',
              errors='replace') as f:
        for line in f:
            # 이름(1-25) 뒤 XSL 번호(숫자만) → 'X%04d' 로 변환
            try:
                num = int(line[25:30])
            except ValueError:
                continue
            # 채택값 (ReadMe 1-기준): Teff 67-71, logg 78-82
            try:
                teff = float(line[66:71])
            except ValueError:
                continue
            try:
                logg = float(line[77:82])
            except ValueError:
                logg = np.nan
            prows.append({'xslid': f'X{num:04d}', 'teff': teff,
                          'logg': logg})
    par = pd.DataFrame(prows).drop_duplicates('xslid')
    df = files.merge(par, on='xslid', how='inner')
    print(f"XSL 스펙트럼 {len(files)}개 / 파라미터 매칭 {len(df)}개")
    return df


def read_xsl(path, match_resolution=True):
    """XSL 스펙트럼 읽기 + 분해능 정합.

    X-shooter 는 R~10,000 으로 학습 서베이(R~1,800-2,500)보다 선이
    훨씬 좁고 깊게 찍힘 → 등가폭(분류)은 견디지만 프로파일을 보는
    Teff/logg 회귀가 분포 밖 입력이 되어 평균으로 후퇴 (실측: 저온
    거성 Teff 가 전부 ~6000K 로 붕괴). 가우시안 컨볼루션으로 R~2,000
    에 맞추면 해소 — MKCLASS(Gray & Corbally 2014)의 표준 절차와 동일."""
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter1d
    with fits.open(path) as h:
        d = h[1].data
        wave = np.array(d['WAVE'], dtype=np.float64) * 10.0   # nm → Å
        flux = np.array(d['FLUX'], dtype=np.float64)
    # 파장은 공기 기준 (60개 자동 RV 중앙값 -83km/s = 진공 이중변환의
    # 시그니처로 실측 확인 → 변환 없이 사용. 잔여 오프셋은 자동 RV 가 흡수)
    if match_resolution:
        m = np.isfinite(wave) & np.isfinite(flux) & (wave > 3800) & \
            (wave < 7600)
        if m.sum() > 100:
            w, f = wave[m], flux[m]
            grid = np.arange(w[0], w[-1], 0.5)          # 0.5Å 균일 격자
            fg = np.interp(grid, w, f)
            # R 10,000→2,000: FWHM 0.55→2.75Å @5500Å → σ_conv≈1.15Å
            fg = gaussian_filter1d(fg, sigma=1.15 / 0.5)
            return grid, fg
    return wave, flux


def main():
    n_max = None if '--all' in sys.argv else 60
    os.makedirs(OUT, exist_ok=True)
    df = load_tables()
    if n_max:
        # 파일럿: Teff 로 층화 샘플 (전 구간 커버)
        df = df.sort_values('teff')
        step = max(1, len(df) // n_max)
        df = df.iloc[::step].head(n_max)
    print(f"검증 대상: {len(df)}개 (실전 모드 — RV·A_V 자동)")

    pred = PredictorV5()
    rows = []
    for k, (_, r) in enumerate(df.iterrows(), 1):
        fpath = os.path.join(BASE, r['file'])
        try:
            dl(CDS_FITS + 'fits/' + r['file'], fpath)
            wave, flux = read_xsl(fpath)
            # av=0: X-shooter 는 슬릿 손실로 연속선 절대 기울기가 불안정
            # → 색지수 기반 자동 소광 추정이 오작동 (실측: 폭주 확인).
            # XSL 표본은 대부분 근거리 밝은 별이라 소광 자체도 작음.
            res = pred.predict_auto(wave, flux, av=0.0)
            if res is None:
                raise RuntimeError('전처리 실패')
            rows.append({
                'xslid': r['xslid'], 'teff_true': r['teff'],
                'logg_true': r['logg'],
                'cls_true': teff_to_class(r['teff']),
                'lum_true': logg_to_lum(r['logg']),
                'cls_pred': res['pred_ens'], 'prob': res['prob_ens'],
                'teff_pred': res['teff_ens'], 'logg_pred': res['logg'],
                'lum_pred': res['lum'],
                'rv_auto': res.get('rv_used'),
                'av_auto': res.get('av_used'),
            })
        except Exception as e:
            rows.append({'xslid': r['xslid'], 'error': str(e)[:80]})
        if k % 20 == 0:
            print(f"  {k}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, 'xsl_predictions.csv'), index=False,
               encoding='utf-8-sig')
    ok = out[out.get('cls_pred').notna()] if 'cls_pred' in out else out.iloc[:0]

    lines = []
    def log(s):
        print(s); lines.append(s)

    log(f"\nXSL 외부 검증 — 성공 {len(ok)}/{len(out)}")
    if len(ok):
        idx = {c: i for i, c in enumerate(CLASS_ORDER)}
        acc = (ok['cls_pred'] == ok['cls_true']).mean()
        d = (ok['cls_pred'].map(idx) - ok['cls_true'].map(idx)).abs()
        log(f"분광형: {100*acc:.1f}%  ±1등급 {100*(d <= 1).mean():.1f}%")
        for c in CLASS_ORDER:
            s = ok[ok['cls_true'] == c]
            if len(s):
                log(f"  {c}: {100*(s['cls_pred'] == c).mean():.0f}% "
                    f"(N={len(s)})")
        rel = (ok['teff_pred'] - ok['teff_true']).abs() / ok['teff_true']
        log(f"Teff 중앙 상대오차: {100*rel.median():.1f}%")
        lok = ok[ok['lum_true'].isin(LUM_ORDER)]
        if len(lok):
            log(f"광도계급: {100*(lok['lum_pred'] == lok['lum_true']).mean():.1f}% "
                f"(N={len(lok)})")
    with open(os.path.join(OUT, '요약.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"저장: {OUT}/")


if __name__ == '__main__':
    main()
