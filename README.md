# 항성 분광형 AI 자동 분류

> 스펙트럼 하나로 별의 분광형(OB·A·F·G·K·M) · 광도계급(거성/주계열/백색왜성) ·
> 유효온도 · 표면중력을 판정하는 딥러닝 프로그램

**🌐 프로젝트 소개 페이지**: https://jigooborn.github.io/nobellsang-batnenyonego/

제72회 전국과학전람회 출품작 (목천고등학교 이한결·맹진호, 지도교원 노민경)

![AI 예측 H-R도](docs/figures/2_hr_diagram.png)

## 성능 (전부 학습 미사용 데이터)

| 검증 | 정확도 |
|---|---|
| 내부 test (16,189개) | **97.6%** (±1등급 99.7%) |
| 5-fold 교차검증 | **97.7 ± 0.06%** |
| 외부 망원경 XSL/VLT (751개) | **94.0%** (±1등급 99.6%) |
| 유효온도 / 표면중력 | 오차 1.14% / 0.152 dex |

## 빠른 시작

```bash
pip install -r requirements.txt
python classify_gui_v5.py samples/spec_401011246.fits
```

동봉된 샘플로 바로 테스트할 수 있습니다 (`samples/`):

| 파일 | 정체 | 볼거리 |
|---|---|---|
| `spec_525402055.fits` | LAMOST 고온성 (실측 30,448K) | OB형 + He II 검출 → "O급 가능성" 표시 |
| `spec_401011246.fits` | LAMOST G형 주계열 | 태양형 스펙트럼, G밴드·Mg b 근거선 |
| `spec_823208221.fits` | LAMOST M형 (3,406K) | TiO 분자띠가 지배하는 저온성 |
| `s0298.fits` | MILES HD 338529 | SIMBAD에 'B5'로 등록된 별 — AI는 F형 판정 (라벨 재검토 사례) |

GUI에서는 파일/폴더 열기, 판정 근거 흡수선 표시, 시선속도·성간소광 자동 추정,
SIMBAD 실시간 대조, 폴더 일괄 분류(CSV)가 가능합니다.

## 저장소 구성

```
classify_gui_v5.py    분류 프로그램 (GUI + CLI)
preprocess_core.py    전처리·피처 추출 모듈
train_v5.py           모델 정의 + 학습 스크립트
models/               학습된 가중치 (CNN 3.4MB + MLP 0.7MB)
data/                 정규화 통계 (추론에 필요)
samples/              테스트용 스펙트럼 4개
pipeline/             연구 전체 재현용 스크립트 (선택)
```

## 원본 데이터 받는 곳

본 저장소는 코드·모델·샘플만 포함합니다. 전체 데이터는 각 기관에서 무료로 받을 수 있습니다.

| 데이터 | 링크 |
|---|---|
| LAMOST DR9 저분해능 스펙트럼 | https://www.lamost.org/dr9/ |
| SDSS MaStar 항성 라이브러리 (DR17) | https://www.sdss4.org/dr17/mastar/ |
| SDSS SEGUE 스펙트럼 | https://www.sdss4.org/dr17/spectro/ |
| MILES 라이브러리 v9.1 | http://miles.iac.es/ |
| XSL DR3 (외부 검증용) | https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/660/A34 |
| Gaia DR3 소광량 (GSP-Phot) | https://gea.esac.esa.int/archive/ |
| SIMBAD | https://simbad.cds.unistra.fr/ |

샘플 FITS의 저작권은 각 서베이(LAMOST DR9, MILES)에 있으며 테스트 용도로만 동봉했습니다.

## 라이선스

MIT License — 코드와 모델 가중치에 적용. 데이터는 각 원 기관의 라이선스를 따릅니다.
