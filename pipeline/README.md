# pipeline/ — 학습·검증 재현용 스크립트

프로그램 사용에는 필요 없고, 연구 전체를 재현할 때만 사용합니다.
원본 스펙트럼(약 15GB)은 README의 데이터 출처에서 받아야 합니다.

순서: build_av_v5 → step1_v5 → prep_dataset_v5 → train_v5(루트) → eval_v5 → kfold_v5 → xsl_validate → simbad_recheck

실행은 저장소 루트에서: `python pipeline/step1_v5.py all` 형태로 하되,
스크립트가 루트의 preprocess_core/train_v5 를 import 하므로
`set PYTHONPATH=.` (Windows) 후 실행하세요.
