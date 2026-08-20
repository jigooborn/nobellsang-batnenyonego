# -*- coding: utf-8 -*-
"""
eval_all_v5.py — 라벨 있는 "모든" 전처리 스펙트럼 전수 검사 (v5)

test 만이 아니라 캡/버퍼로 제외된 것까지 전부 분류해 숨은 서베이-클래스
편중을 찾는다 (v3 때 이 검사로 "LAMOST F 50% 붕괴"를 발견했음 — 함정 2번).

  train  : 학습에 쓰인 것 (과장됨, 참고용)
  test   : 공식 성능
  unused : 라벨은 있지만 학습 제외 (준-독립 검증)

실행 (v5 폴더): python eval_all_v5.py
출력: results/eval_all_v5/ (혼동행렬 png, predictions csv)
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import torch

from preprocess_core import CLASS_ORDER, N_CLASSES, teff_to_class
from train_v5 import SpectralResNetV5, FeatureMLPV5, DEVICE

SHARD_DIR = "outputs_preprocessed_v5"
CATALOG   = "master_catalog_v5.csv"
OUT_DIR   = os.path.join("results", "eval_all_v5")
BATCH     = 1024
CLS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


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
    raise ValueError(survey)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cat = pd.read_csv(CATALOG, usecols=['spec_id', 'source', 'teff_adopted'])
    cat = cat.drop_duplicates('spec_id')

    tr = pd.read_csv("data/v5_train_labels.csv", usecols=['STAR_ID'])
    te = pd.read_csv("data/v5_test_labels.csv", usecols=['STAR_ID'])
    train_ids = set(s.split('#')[0] for s in tr['STAR_ID'])
    test_ids = set(te['STAR_ID'])

    fstats = np.load("data/feat_norm_v5.npy")
    f_mu, f_sd = (fstats[0].astype(np.float32),
                  fstats[1].astype(np.float32))

    cnn = SpectralResNetV5().to(DEVICE)
    cnn.load_state_dict(torch.load("models/resnet_v5.pth",
                                   map_location=DEVICE, weights_only=True))
    cnn.eval()
    mlp = FeatureMLPV5().to(DEVICE)
    mlp.load_state_dict(torch.load("models/mlp_v5.pth",
                                   map_location=DEVICE, weights_only=True))
    mlp.eval()

    rows_out = []
    for mp in sorted(glob.glob(os.path.join(SHARD_DIR, "*_meta.csv"))):
        tag = os.path.basename(mp)[:-len("_meta.csv")]
        survey = tag.split('_shard')[0]
        meta = pd.read_csv(mp)
        ok = meta['status'] == 'ok'
        df = pd.DataFrame({'key': make_keys(survey, meta),
                           'flux_row': meta['flux_row']})[ok.values]
        df = df.merge(cat, left_on='key', right_on='spec_id', how='inner')
        t = df['teff_adopted'].values
        df = df[np.isfinite(t) & (t > 1000) & (t < 60000)]
        if not len(df):
            continue
        print(f"[{tag}] 라벨 보유 {len(df):,}개 분류 중...", flush=True)

        flux = np.load(os.path.join(SHARD_DIR, f"{tag}_flux.npy"),
                       mmap_mode='r')
        feat = np.load(os.path.join(SHARD_DIR, f"{tag}_features.npy"),
                       mmap_mode='r')
        rows = df['flux_row'].values.astype(int)
        labels = np.array([CLS_TO_IDX[teff_to_class(x)]
                           for x in df['teff_adopted'].values])

        preds = np.empty(len(df), dtype=int)
        probs = np.empty(len(df), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(df), BATCH):
                r = rows[s:s + BATCH]
                fb = torch.from_numpy(np.ascontiguousarray(flux[r])
                                      ).float().to(DEVICE)
                xb = (np.ascontiguousarray(feat[r]) - f_mu) / f_sd
                xb = torch.from_numpy(xb.astype(np.float32)).to(DEVICE)
                lc, _, _, _ = cnn(fb, xb)
                lm, _, _, _ = mlp(fb, xb)
                pe = (torch.softmax(lc, 1) + torch.softmax(lm, 1)) / 2
                pm_, pi = pe.max(1)
                preds[s:s + BATCH] = pi.cpu().numpy()
                probs[s:s + BATCH] = pm_.cpu().numpy()

        member = np.where(df['spec_id'].isin(test_ids), 'test',
                 np.where(df['spec_id'].isin(train_ids), 'train', 'unused'))
        rows_out.append(pd.DataFrame({
            'spec_id': df['spec_id'].values, 'survey': survey,
            'label': labels, 'pred': preds, 'prob': probs,
            'member': member,
        }))

    allr = pd.concat(rows_out, ignore_index=True)
    allr['label_c'] = [CLASS_ORDER[i] for i in allr['label']]
    allr['pred_c'] = [CLASS_ORDER[i] for i in allr['pred']]
    allr.to_csv(os.path.join(OUT_DIR, "predictions_all_v5.csv"), index=False)

    lines = []
    def report(sub, name):
        if not len(sub):
            return None
        acc = (sub['label'] == sub['pred']).mean()
        adj = ((sub['label'] - sub['pred']).abs() <= 1).mean()
        msg = (f"\n=== {name}: {len(sub):,}개 | 정확도 {acc*100:.2f}% | "
               f"±1등급 {adj*100:.2f}%")
        print(msg); lines.append(msg)
        rec = []
        for i, c in enumerate(CLASS_ORDER):
            m = sub['label'] == i
            rec.append(f"{c}:{(sub['pred'][m] == i).mean()*100:.0f}%"
                       if m.any() else f"{c}:-")
        msg = "  재현율: " + "  ".join(rec)
        print(msg); lines.append(msg)
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for l, p in zip(sub['label'], sub['pred']):
            cm[l, p] += 1
        return cm

    cm_all = report(allr, "전체 (train 포함 — 참고용)")
    report(allr[allr['member'] == 'train'], "train")
    cm_te = report(allr[allr['member'] == 'test'], "test (공식)")
    report(allr[allr['member'] == 'unused'], "unused (준-독립)")
    for sv in sorted(allr['survey'].unique()):
        report(allr[allr['survey'] == sv], f"서베이: {sv}")
    # 서베이×클래스 편중 감시 (함정 2번의 조기 경보)
    msg = "\n[서베이×클래스 재현율 — 50%대가 보이면 편중 의심]"
    print(msg); lines.append(msg)
    for sv in sorted(allr['survey'].unique()):
        sub = allr[allr['survey'] == sv]
        rec = []
        for i, c in enumerate(CLASS_ORDER):
            m = sub['label'] == i
            rec.append(f"{c}:{(sub['pred'][m] == i).mean()*100:.0f}"
                       if m.sum() >= 20 else f"{c}:-")
        msg = f"  {sv:8s} " + " ".join(rec)
        print(msg); lines.append(msg)

    with open(os.path.join(OUT_DIR, "전수검사_v5.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.rc('font', family='Malgun Gothic')
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [(axes[0], cm_all, f"전체 {len(allr):,}개"),
                          (axes[1], cm_te, "test (공식 성능)")]:
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                        fontsize=7,
                        color='white' if norm[i, j] > 0.5 else 'black')
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_ORDER)
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("예측"); ax.set_ylabel("정답")
        ax.set_title(title, fontsize=11)
    fig.suptitle("v5 앙상블 혼동행렬 (6클래스)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "혼동행렬_전체.png"), dpi=140)
    print(f"\n저장: {OUT_DIR}/")


if __name__ == "__main__":
    main()
