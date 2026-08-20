# -*- coding: utf-8 -*-
"""
download_m.py — m_targets.csv 의 LAMOST 스펙트럼 병렬 다운로드

- 4스레드 + 요청 간 짧은 대기 (서버 예의)
- 이어받기: 이미 있는 파일 건너뜀 / .part 원자적 저장
- 실패는 3회 재시도 후 기록만 하고 진행
저장: 스펙트럼원본/lamost(중국)/lamost_spectra/spec_<obsid>.fits
      (기존 폴더 — step1_v5 lamost 재실행 시 자동 포함)
"""
import os
import time
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASE = r'C:\Users\user\Desktop\최종'
TARGETS = os.path.join(BASE, 'v5', 'm_targets.csv')
SAVE_DIR = os.path.join(BASE, '스펙트럼원본', 'lamost(중국)',
                        'lamost_spectra')
URL = 'http://www.lamost.org/dr9/v2.0/spectrum/fits/{obsid}?token='
WORKERS = 4
UA = {'User-Agent': 'Mozilla/5.0'}


def fetch(obsid):
    path = os.path.join(SAVE_DIR, f'spec_{obsid}.fits')
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return 'skip'
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL.format(obsid=obsid),
                                         headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            if len(data) < 1000:
                raise RuntimeError('빈 응답')
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
            time.sleep(0.05)
            return 'ok'
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return 'fail'


def main():
    df = pd.read_csv(TARGETS)
    obsids = df['obsid'].astype(int).tolist()
    print(f"대상 {len(obsids):,}개, {WORKERS}스레드", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(fetch, obsids), 1):
            if res == 'ok': n_ok += 1
            elif res == 'skip': n_skip += 1
            else: n_fail += 1
            if i % 500 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                left = (len(obsids) - i) / max(rate, 1e-9)
                print(f"  {i:,}/{len(obsids):,} ({rate:.1f}개/초, "
                      f"남은 예상 {left/60:.0f}분) "
                      f"성공 {n_ok:,} 실패 {n_fail:,}", flush=True)
    print(f"완료 — 성공 {n_ok:,} / 건너뜀 {n_skip:,} / 실패 {n_fail:,}")


if __name__ == '__main__':
    main()
