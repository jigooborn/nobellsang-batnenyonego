# -*- coding: utf-8 -*-
"""발표용 탁상 바인더 10쪽 — A4 세로, 인쇄용 밝은 바탕.

이미 넣은 '전처리 9단계'와 '분광형 대표 스펙트럼' 뒤에 이어지는 구성.
모든 수치는 검증된_수치.md 의 값(원본 데이터에서 직접 재계산)만 사용한다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from preprocess_core import CLASS_ORDER, N_CLASSES, FEATURE_NAMES

OUT = r'C:\Users\user\Desktop\최종\발표_바인더'
os.makedirs(OUT, exist_ok=True)

BG, INK, MID, FAINT, RULE = '#ffffff', '#15181f', '#5b6270', '#a8aeb9', '#d8dce3'
ACC, WARN, OK = '#1f5fd0', '#c0392b', '#1e8a5f'
CLASS_C = {'OB': '#3f6fd1', 'A': '#7c9ad6', 'F': '#d6a520',
           'G': '#e08a1e', 'K': '#d2601f', 'M': '#b8331f'}

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'figure.facecolor': BG, 'savefig.facecolor': BG, 'axes.facecolor': BG,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MID, 'ytick.color': MID,
    'axes.edgecolor': '#c2c8d2', 'grid.color': '#e6e9ee', 'font.size': 12,
})

W_IN, H_IN, DPI = 8.27, 11.69, 200        # A4 세로
L, R = .075, .925                          # 좌우 여백


def page(num, section, title, sub=None):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.text(L, .967, '   '.join(section), fontsize=10.5, weight='bold', color=ACC)
    fig.text(R, .967, f'{num:02d}', fontsize=11.5, weight='bold',
             color=FAINT, ha='right')
    fig.add_artist(plt.Line2D([L, R], [.954, .954], color=RULE, lw=1))
    fig.text(L, .938, title, fontsize=24, weight='bold', va='top')
    if sub:
        fig.text(L, .895, sub, fontsize=12, color=MID, va='top', linespacing=1.7)
    return fig


def takeaway(fig, text, color=ACC):
    fig.add_artist(Rectangle((L, .048), R - L, .062, transform=fig.transFigure,
                             facecolor='#f2f5fa', edgecolor=color, lw=1.3, zorder=0))
    fig.add_artist(Rectangle((L, .048), .005, .062, transform=fig.transFigure,
                             facecolor=color, lw=0, zorder=1))
    fig.text(L + .022, .079, text, fontsize=11.5, va='center', color=INK,
             linespacing=1.65)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(' ', name)


pred = pd.read_csv('results/eval_v5/test_predictions.csv')
lab = pd.read_csv('data/v5_test_labels.csv')
feat = np.load('data/v5_test_features.npy')
fi = {n: i for i, n in enumerate(FEATURE_NAMES)}

# ══ 01 표지 ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                         facecolor='#0f1218', zorder=-1))
fig.text(.5, .84, '항성 분광형 자동 분류 AI', fontsize=15, color='#8b93a4',
         ha='center', weight='bold')
fig.text(.5, .775, '피커링의 두 눈,', fontsize=40, color='#ffffff',
         ha='center', weight='bold')
fig.text(.5, .705, 'AI 로 재현하다', fontsize=40, color='#6ea8ff',
         ha='center', weight='bold')
fig.add_artist(plt.Line2D([.30, .70], [.664, .664], color='#39404e', lw=1.2))
fig.text(.5, .628, '스펙트럼 한 장으로 분광형 · 광도계급 ·\n'
                   '표면온도 · 표면중력을 판정하고,\n'
                   '한 세기 축적된 SIMBAD 분광형을 독립적으로 재검증하다',
         fontsize=12.5, color='#b9c0cd', ha='center', va='top', linespacing=2.0)

nums = [('97.6%', '학습에 쓰지 않은 별\n16,189개 정확도'),
        ('97.7%', '5-fold 교차검증\n(표준편차 0.06)'),
        ('94.0%', '다른 망원경(VLT)\n외부 검증'),
        ('5,770', 'SIMBAD 분광형과\n대조한 별')]
for i, (n, t) in enumerate(nums):
    x = .29 + (i % 2) * .42
    y = .44 - (i // 2) * .155
    fig.text(x, y, n, fontsize=31, color='#ffffff', ha='center', weight='bold')
    fig.text(x, y - .048, t, fontsize=10.5, color='#8b93a4', ha='center',
             va='top', linespacing=1.8)

fig.text(.5, .115, '제72회 전국과학전람회', fontsize=12, color='#6b7383', ha='center')
fig.text(.5, .088, '목천고등학교  이한결 · 맹진호   ·   지도교원 노민경',
         fontsize=12, color='#6b7383', ha='center')
save(fig, '01_표지.png')

# ══ 02 동기와 목적 ════════════════════════════════════════════════
fig = page(2, 'MOTIVATION', '왜 다시 분류하는가',
           '1890년대 하버드 천문대에서 피커링과 캐넌은 사진건판에 찍힌 별빛을\n'
           '눈으로 보고 분류했다. 그들이 본 두 가지 — 스펙트럼 전체의 인상과\n'
           '개별 흡수선의 세기 — 를 두 개의 AI 모델에 나누어 맡겼다.')

boxes = [('문제 1', '사람이 일일이 볼 수 없는 양',
          '오늘날 서베이 망원경은 하룻밤에 수천 개의 스펙트럼을 쏟아낸다.\n'
          '눈으로 분류하던 방식으로는 따라갈 수 없다.', '#3f6fd1'),
         ('문제 2', '기록마다 기준과 시대가 다르다',
          'SIMBAD 의 분광형은 1890년대 사진건판부터 최근 관측까지\n'
          '뒤섞여 있다. 같은 별에 문헌마다 다른 값이 등록된 경우도 있다.', '#d2601f'),
         ('목적', '독립된 눈으로 다시 본다',
          '스펙트럼만 보고 판정하는 AI 를 만들어 기존 라벨을 참고하지 않고\n'
          '재판정한 뒤, 어긋난 별을 제3의 물리량으로 심판한다.', '#1e8a5f')]
for i, (tag, head, body, c) in enumerate(boxes):
    y = .745 - i * .195
    fig.add_artist(Rectangle((L, y - .155), R - L, .155, transform=fig.transFigure,
                             facecolor='#fbfcfe', edgecolor=c, lw=1.5))
    fig.add_artist(Rectangle((L, y - .155), .006, .155, transform=fig.transFigure,
                             facecolor=c, lw=0))
    fig.text(L + .025, y - .022, tag, fontsize=10.5, color=c, weight='bold', va='top')
    fig.text(L + .025, y - .052, head, fontsize=15, weight='bold', va='top')
    fig.text(L + .025, y - .095, body, fontsize=11.5, color=MID, va='top',
             linespacing=1.8)

takeaway(fig, '기존 라벨을 학습에 전혀 쓰지 않았기 때문에,\nAI 의 판정은 그 라벨을 검증하는 독립된 잣대가 된다.')
save(fig, '02_연구동기와목적.png')

# ══ 03 데이터 구축 ════════════════════════════════════════════════
fig = page(3, 'DATA', '신뢰할 수 있는 학습셋 만들기',
           '많이 모으는 것보다 중요한 것은 정답이 새지 않게 만드는 일이다.\n'
           '같은 별이 학습과 시험에 나뉘어 들어가면 성적이 부풀려지므로,\n'
           '위치로 같은 별을 묶어 통째로 갈랐다.')

steps = [('01  수집·전처리', '273,403', '스펙트럼',
          'LAMOST 182,293 · MaStar 59,266\nSEGUE 30,859 · MILES 985', '#3f6fd1'),
         ('02  통합 정답지', '222,381', '행',
          'Gaia DR3 로 같은 별을 묶어 그룹화\n— 학습·시험 누수를 원천 차단', '#1e8a5f'),
         ('03  학습', '109,670', '표본',
          '원본 92,157 + 증강 17,513\n증강은 학습에만 적용', '#d6a520'),
         ('04  시험', '16,189', '표본',
          '학습에 한 번도 등장하지 않은 별\n(전체 83,435개 별 그룹을 분할)', '#d2601f')]
for i, (tag, num, unit, body, c) in enumerate(steps):
    y = .755 - i * .158
    fig.add_artist(Rectangle((L, y - .125), R - L, .125, transform=fig.transFigure,
                             facecolor=c + '10', edgecolor=c, lw=1.4))
    fig.text(L + .022, y - .022, tag, fontsize=11.5, color=c, weight='bold', va='top')
    fig.text(L + .022, y - .095, body, fontsize=11, color=MID, va='top',
             linespacing=1.8)
    fig.text(R - .025, y - .048, num, fontsize=26, color=c, weight='bold',
             ha='right', va='center')
    fig.text(R - .025, y - .092, unit, fontsize=10.5, color=MID, ha='right', va='center')
    if i < 3:
        fig.add_artist(FancyArrowPatch((.5, y - .128), (.5, y - .152),
                                       transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=15,
                                       color='#9aa2b0', lw=1.8))

takeaway(fig, '별 그룹 단위로 나누었기 때문에 시험에 쓴 16,189개는\n학습 과정에서 한 번도 등장하지 않는다.')
save(fig, '03_데이터구축.png')

# ══ 04 AI 구조 ════════════════════════════════════════════════════
fig = page(4, 'MODEL', '두 개의 눈으로 본다',
           '스펙트럼 전체를 보는 눈과 물리 지표만 보는 눈,\n둘의 판정을 합쳐 최종 결정한다.')

eyes = [('EYE 01', 'CNN (ResNet 구조)',
         '정규화된 스펙트럼 3,401개 화소를 통째로 입력한다.\n'
         '사람이 정해주지 않은 특징까지 스스로 찾아낸다.',
         ['3,401 화소 전체', 'SE 어텐션', '어텐션 풀링'], '#3f6fd1'),
        ('EYE 02', 'MLP (물리 지표 39개)',
         '물리 법칙에서 유도한 값만 입력한다. 무엇을 보고\n'
         '판단했는지가 사람의 언어로 정의되어 있다.',
         ['등가폭 27개', '선폭 10개', '색지수 2개'], '#d2601f')]
for i, (tag, head, body, tg, c) in enumerate(eyes):
    y = .800 - i * .196
    fig.add_artist(Rectangle((L, y - .176), R - L, .176, transform=fig.transFigure,
                             facecolor='#fbfcfe', edgecolor=c, lw=1.5))
    fig.text(L + .022, y - .022, tag, fontsize=10.5, color=c, weight='bold', va='top')
    fig.text(L + .022, y - .052, head, fontsize=15, weight='bold', va='top')
    fig.text(L + .022, y - .093, body, fontsize=11.5, color=MID, va='top',
             linespacing=1.8)
    for j, t in enumerate(tg):
        bx = L + .022 + j * .175
        fig.add_artist(Rectangle((bx, y - .168), .162, .028,
                                 transform=fig.transFigure,
                                 facecolor='#ffffff', edgecolor='#ccd2dc', lw=1))
        fig.text(bx + .081, y - .154, t, fontsize=9.5, color=MID,
                 ha='center', va='center')

fig.text(L, .392, '물리 지표가 근거로 삼는 이론', fontsize=13, weight='bold', va='top')
theory = [('빈의 변위 법칙', '색지수 2개', '뜨거울수록 파란빛이 강하다'),
          ('사하-볼츠만 식', '등가폭 27개', '원소마다 흡수선이 최대가 되는 온도가 다르다'),
          ('압력 넓어짐', '선폭 10개', '표면중력이 크면 흡수선이 넓어진다')]
for i, (law, ind, why) in enumerate(theory):
    y = .352 - i * .052
    fig.text(L + .008, y, law, fontsize=11.5, weight='bold', va='center', color=ACC)
    fig.text(L + .20, y, ind, fontsize=11.5, va='center')
    fig.text(L + .345, y, why, fontsize=11, va='center', color=MID)

fig.add_artist(Rectangle((L, .155), R - L, .045, transform=fig.transFigure,
                         facecolor='#f7f9fc', edgecolor='#ccd2dc', lw=1))
fig.text(L + .022, .1775, '두 판정이 엇갈리거나 확신이 낮으면 → 특이 천체 가능성으로 표시해 사람에게 넘긴다',
         fontsize=11, va='center', color=INK)

takeaway(fig, '두 모델이 서로 다른 근거를 보기 때문에,\n한쪽이 헷갈리는 별을 다른 쪽이 잡아낸다.')
save(fig, '04_AI구조.png')

# ══ 05 혼동행렬 ═══════════════════════════════════════════════════
idx = {c: i for i, c in enumerate(CLASS_ORDER)}
y_ = pred['cls_true'].map(idx).values
p_ = pred['cls_pred'].map(idx).values
cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for a, b in zip(y_, p_):
    cm[a, b] += 1
norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
acc, adj = (y_ == p_).mean(), (np.abs(y_ - p_) <= 1).mean()

fig = page(5, 'ACCURACY', f'정확도 {100*acc:.1f}%',
           f'학습에 한 번도 쓰지 않은 별 {len(pred):,}개로 측정했다.\n'
           f'틀린 경우까지 포함해 {100*adj:.1f}% 가 한 등급 이내에 들어온다.')
ax = fig.add_axes([.20, .43, .62, .40])
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cfe0ff', '#5b8ede', '#1f4fa8'])
ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if cm[i, j] == 0:
            continue
        ax.text(j, i, f'{cm[i, j]:,}\n{100*norm[i, j]:.1f}%', ha='center', va='center',
                fontsize=8.5, weight='bold' if i == j else 'normal',
                color='#ffffff' if norm[i, j] > .5 else INK)
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_ORDER, fontsize=13)
ax.set_yticks(range(N_CLASSES))
ax.set_yticklabels([f'{c}  {100*norm[k, k]:.1f}%' for k, c in enumerate(CLASS_ORDER)],
                   fontsize=11)
ax.set_xlabel('AI 판정', fontsize=12); ax.set_ylabel('정답 (재현율)', fontsize=12)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

fig.text(L, .385, '읽는 법', fontsize=13, weight='bold', va='top')
fig.text(L, .348,
         '대각선이 맞힌 별이다. 틀린 경우도 거의 모두 바로 옆 등급에 몰려 있는데,\n'
         '온도가 이웃한 분광형은 경계가 연속이기 때문이다.\n'
         '완전히 빗나간 판정은 사실상 없다.',
         fontsize=11.5, color=MID, va='top', linespacing=1.85)

fig.text(L, .245, '함께 예측하는 물리량', fontsize=13, weight='bold', va='top')
regs = [('유효온도', '중앙 상대오차 1.14%'),
        ('표면중력', '평균절대오차 0.152 dex'),
        ('광도계급 3분류', '96.8%  (거성 92.1 / 주계열 98.8 / 백색왜성 30.2)')]
for i, (a, b) in enumerate(regs):
    yy = .208 - i * .038
    fig.text(L + .008, yy, a, fontsize=11.5, weight='bold', va='center', color=ACC)
    fig.text(L + .215, yy, b, fontsize=11.5, va='center', color=MID)

takeaway(fig, '틀려도 이웃 등급 안에 머문다는 것은\n온도 순서를 제대로 배웠다는 뜻이다.')
save(fig, '05_정확도_혼동행렬.png')

# ══ 06 검증 ═══════════════════════════════════════════════════════
fig = page(6, 'VALIDATION', '세 단계로 검증했다',
           '한 번의 시험은 우연일 수 있다. 나누는 방법을 바꾸고,\n'
           '마지막에는 학습에 없던 다른 망원경의 관측으로 시험했다.')
ax = fig.add_axes([.14, .50, .72, .32])
names = ['홀드아웃 test\n16,189개', '5-fold 교차검증\n다섯 번 재학습', 'XSL 외부 (VLT)\n751개']
vals = [97.59, 97.73, 94.0]
errs = [0, 0.06, 0]
cols = ['#3f6fd1', '#2e7fd0', '#1e8a5f']
bars = ax.bar(names, vals, color=cols, width=.55, alpha=.9, yerr=errs,
              capsize=6, ecolor=MID)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .45, f'{v:.1f}%', ha='center',
            fontsize=15, weight='bold')
ax.set_ylim(90, 100); ax.set_ylabel('정확도 (%)', fontsize=12)
ax.tick_params(axis='x', labelsize=10.5)
ax.grid(axis='y', alpha=.5); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.text(L, .445, '각 단계가 확인하는 것', fontsize=13, weight='bold', va='top')
rows = [('홀드아웃 test', '학습에 쓰지 않은 별로 기본 성능을 잰다', '97.59%  ·  ±1등급 99.68%'),
        ('5-fold 교차검증', '데이터를 다섯 번 다르게 갈라도 흔들리지 않는가',
         '97.73 ± 0.06%  (CNN 97.80 / MLP 96.76)'),
        ('XSL 외부 (VLT)', '학습에 없던 다른 망원경에서도 되는가',
         '94.0%  ·  ±1등급 99.6%  ·  광도계급 92.7%')]
for i, (a, b, c) in enumerate(rows):
    yy = .425 - i * .082
    fig.text(L + .008, yy, a, fontsize=12, weight='bold', va='top', color=ACC)
    fig.text(L + .008, yy - .028, b, fontsize=11, color=MID, va='top')
    fig.text(L + .008, yy - .053, c, fontsize=11, va='top')

fig.add_artist(Rectangle((L, .128), R - L, .05, transform=fig.transFigure,
                         facecolor='#f7f9fc', edgecolor='#ccd2dc', lw=1))
fig.text(L + .022, .153, 'XSL 은 VLT/X-shooter 관측이다. 분해능을 맞춘 뒤 측정했다.',
         fontsize=11, va='center', color=INK)

takeaway(fig, '세 단계 모두 학습에 쓰지 않은 별로만 측정했다.')
save(fig, '06_검증.png')

# ══ 07 사하-볼츠만 ════════════════════════════════════════════════
teff = lab['TEFF'].values
show = [('ew_Hbeta', 'Hβ 수소선', '#3f6fd1'), ('ew_He4471', 'He I 헬륨선', '#7c52ab'),
        ('ew_Ca4227', 'Ca I 칼슘선', '#1e8a5f'), ('ew_Mgb', 'Mg b 마그네슘', '#d2601f'),
        ('ew_TiO2', 'TiO 분자띠', '#b8331f')]
bins = np.logspace(np.log10(2800), np.log10(35000), 24)
fig = page(7, 'EVIDENCE 01', '원소선은 각자의 온도에서만 최대가 된다',
           'AI 가 실제로 읽어낸 흡수선 세기를 온도순으로 늘어놓자,\n'
           '교과서의 사하-볼츠만 이론이 예측한 봉우리가 그대로 나타났다.')
ax = fig.add_axes([.135, .45, .79, .37])
UNREL = 4000.0
for key, name, c in show:
    v = feat[:, fi[key]]
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (teff >= bins[i]) & (teff < bins[i + 1])
        if m.sum() >= 15:
            xs.append(np.sqrt(bins[i] * bins[i + 1])); ys.append(np.median(v[m]))
    xs, ys = np.array(xs), np.array(ys)
    if key in ('ew_Hbeta', 'ew_He4471'):
        hot = xs >= UNREL
        ax.plot(xs[hot], ys[hot], 'o-', color=c, lw=2.2, ms=4, label=name)
        cool = xs <= UNREL
        j = np.where(hot)[0]
        if j.size:
            cool[j[0]] = True
        ax.plot(xs[cool], ys[cool], ':', color=c, lw=1.8, alpha=.55)
    else:
        ax.plot(xs, ys, 'o-', color=c, lw=2.2, ms=4, label=name)
ax.axvspan(2700, UNREL, color='#000000', alpha=.045, zorder=0)
ax.set_xscale('log'); ax.set_xlim(36000, 2700)
ylo, ymx = ax.get_ylim(); ax.set_ylim(ylo, ymx * 1.22)
tk = [30000, 10000, 7000, 5000, 3500]
ax.set_xticks(tk); ax.set_xticklabels([f'{t:,}' for t in tk]); ax.set_xticks([], minor=True)
for b in [10000, 7500, 6000, 5000, 3500]:
    ax.axvline(b, color='#dfe3ea', ls=':', lw=1)
for c_, xm in [('OB', 19000), ('A', 8700), ('F', 6700), ('G', 5480),
               ('K', 4180), ('M', 3100)]:
    ax.text(xm, ymx * 1.14, c_, fontsize=12.5, weight='bold', color=CLASS_C[c_],
            ha='center', va='center')
ax.set_xlabel('표면온도 (K)   ← 높음', fontsize=12)
ax.set_ylabel('흡수선 세기 (등가폭, Å)', fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(.01, .88), fontsize=10)
ax.grid(alpha=.45); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.text(L, .40, '측정된 봉우리가 문헌값과 일치한다', fontsize=13, weight='bold', va='top')
peaks = [('수소  Hβ', '9,074 K', '문헌 A0형 약 9,500 K'),
         ('헬륨  He I', '19,871 K', '문헌 B2형 약 20,000 K')]
for i, (a, b, c) in enumerate(peaks):
    yy = .358 - i * .045
    fig.text(L + .008, yy, a, fontsize=11.5, weight='bold', va='center', color=ACC)
    fig.text(L + .17, yy, b, fontsize=11.5, va='center')
    fig.text(L + .32, yy, c, fontsize=11, va='center', color=MID)
fig.text(L + .008, .265,
         'Mg · Ca 는 K형에서, TiO 는 분자가 살아남는 3천 K대에서만 강해진다.',
         fontsize=11.5, color=MID, va='top')

fig.add_artist(Rectangle((L, .155), R - L, .075, transform=fig.transFigure,
                         facecolor='#fdf6f5', edgecolor='#e8c4bf', lw=1))
fig.text(L + .022, .213, '음영 구간의 점선은 물리가 아니라 측정의 한계다',
         fontsize=11.5, weight='bold', va='top', color=WARN)
fig.text(L + .022, .184, 'TiO 분자띠가 유사연속선을 눌러 H · He 등가폭이 과대평가된다.',
         fontsize=11, color=MID, va='top')

takeaway(fig, '봉우리 온도가 문헌값과 거의 일치한다.\n데이터를 외운 것이 아니라 원소의 물리를 읽어냈다는 뜻이다.')
save(fig, '07_사하볼츠만.png')

# ══ 08 Kiel 도 ════════════════════════════════════════════════════
fig = page(8, 'EVIDENCE 02', 'AI 의 답만으로 진화 구조가 드러났다',
           '문헌값은 한 개도 넣지 않았다. AI 가 스펙트럼만 보고 답한\n'
           '온도와 표면중력으로 점을 찍었을 뿐이다.')
ax = fig.add_axes([.145, .40, .78, .42])
for c in CLASS_ORDER:
    m = pred['cls_pred'] == c
    ax.scatter(pred.loc[m, 'teff_pred'], pred.loc[m, 'logg_pred'], s=3, alpha=.35,
               c=CLASS_C[c], label=f'{c}형', edgecolors='none', rasterized=True)
ax.set_xscale('log'); ax.set_xlim(45000, 2700); ax.set_ylim(5.6, 0)
ax.set_xticks([30000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(['30,000', '10,000', '6,000', '4,000', '3,000'])
ax.set_xticks([], minor=True)
ax.set_xlabel('AI 예측 표면온도 (K)   ← 높음', fontsize=12)
ax.set_ylabel('AI 예측 표면중력 logg', fontsize=12)
ax.annotate('주계열 (왜성)', xy=(6300, 4.6), fontsize=12.5, weight='bold', ha='center')
ax.annotate('적색거성가지', xy=(4150, 1.8), fontsize=12.5, weight='bold',
            color='#b8331f', ha='center')
lg = ax.legend(markerscale=5, loc='lower left', ncol=3, fontsize=9.5, framealpha=.95)
for lh in lg.legend_handles:
    lh.set_alpha(1)
ax.grid(alpha=.4); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.text(L, .355, '교과서 H-R도와 무엇이 다른가', fontsize=13, weight='bold', va='top')
fig.text(L, .315,
         '세로축이 광도가 아니라 표면중력이다. 광도를 구하려면 별까지의 거리를 알아야\n'
         '하는데, 스펙트럼 한 장에는 거리 정보가 없다. 그래서 분광 관측에서는\n'
         '표면중력을 세로축에 쓰는 Kiel 도(분광 H-R도)를 쓴다.\n\n'
         '교과서에서 비스듬히 내려가던 주계열이 여기서는 가로로 누운 띠로 보인다.\n'
         '축이 다를 뿐, 담고 있는 물리는 같다.',
         fontsize=11.5, color=MID, va='top', linespacing=1.85)

takeaway(fig, '주계열과 적색거성가지가 저절로 갈라져 나타났다.\n두 축 모두 AI 가 스펙트럼만 보고 답한 값이다.')
save(fig, '08_Kiel도.png')

# ══ 09 SIMBAD 재검증 ══════════════════════════════════════════════
fig = page(9, 'RECHECK', '100년의 기록을 오늘의 눈으로',
           '학습에 쓰지 않은 별 가운데 SIMBAD 에 분광형이 등록된\n'
           '5,770개를 대조했다. 일치율 71.5%, 한 등급 이내 99.1%.')

grades = [('B', 9, 100.0), ('C', 1100, 85.9), ('D', 2888, 74.3), ('E', 1752, 57.3)]
ax = fig.add_axes([.145, .555, .78, .265])
gc = ['#1e8a5f', '#5b9bd5', '#d6a520', '#c0392b']
bars = ax.bar([g[0] for g in grades], [g[2] for g in grades], color=gc,
              width=.55, alpha=.9)
for b, (g, n, v) in zip(bars, grades):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f'{v:.1f}%', ha='center',
            fontsize=13, weight='bold')
    ax.text(b.get_x() + b.get_width() / 2, 5, f'{n:,}개', ha='center',
            fontsize=10, color='#ffffff', weight='bold')
ax.set_ylim(0, 115)
ax.set_xlabel('SIMBAD 가 스스로 매긴 라벨 품질등급   (A 최고 → E 최저)', fontsize=11.5)
ax.set_ylabel('AI 와 일치한 비율 (%)', fontsize=11.5)
ax.grid(axis='y', alpha=.45); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.add_artist(Rectangle((L, .445), R - L, .07, transform=fig.transFigure,
                         facecolor='#eef4ff', edgecolor=ACC, lw=1.3))
fig.text(L + .022, .498, '핵심 발견', fontsize=12, weight='bold', va='top', color=ACC)
fig.text(L + .022, .472,
         'AI 가 아무 데서나 어긋나는 것이 아니다. SIMBAD 스스로 신뢰도가 낮다고\n'
         '표시한 라벨일수록 정확히 그만큼 더 어긋난다.',
         fontsize=11, va='top', linespacing=1.7)

fig.text(L, .415, '불일치 1,645개를 서베이 실측 온도로 심판', fontsize=13,
         weight='bold', va='top')
judge = [('1,570개', '실측 온도가 AI 편', OK),
         ('61개', '실측 온도가 SIMBAD 편 — AI 오분류', WARN),
         ('14개', '제3의 값', MID)]
for i, (n, t, c) in enumerate(judge):
    yy = .375 - i * .04
    fig.text(L + .012, yy, n, fontsize=12, weight='bold', color=c, va='center')
    fig.text(L + .14, yy, t, fontsize=11.5, color=MID, va='center')

fig.add_artist(Rectangle((L, .155), R - L, .09, transform=fig.transFigure,
                         facecolor='#fbfcfe', edgecolor='#ccd2dc', lw=1.2))
fig.text(L + .022, .229, '재검토 후보 1,225개  —  그중 두 등급 이상 어긋난 강한 후보 34개',
         fontsize=12, weight='bold', va='top')
fig.text(L + .022, .198,
         'HD 275900   SIMBAD "F5" (1995년, 등급 E)  →  AI 는 OB형 96%\n'
         '               서베이 실측 온도 11,809 K = OB형',
         fontsize=10, color=MID, va='top', linespacing=1.7)

takeaway(fig, '재검토 후보는 오류 확정이 아니라 후보 선별이다.\n최종 판단은 원 관측 자료를 가진 연구자의 몫이다.')
save(fig, '09_SIMBAD재검증.png')

# ══ 10 한계와 전망 ════════════════════════════════════════════════
feh = lab['FEH'].values
wrong = (y_ != p_)
edges = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.6]
lbls = ['< -1.5', '-1.5~-1.0', '-1.0~-0.5', '-0.5~0.0', '> 0.0']
rates, ns = [], []
for i in range(len(edges) - 1):
    m = np.isfinite(feh) & (feh >= edges[i]) & (feh < edges[i + 1])
    rates.append(100 * wrong[m].mean() if m.sum() else 0); ns.append(int(m.sum()))

fig = page(10, "WHAT'S NEXT", '아직 남은 과제',
           '잘 맞힌 곳보다 틀린 곳이 다음 연구의 방향을 알려준다.\n'
           '어디서 왜 어려웠는지를 물리적으로 짚었다.')

ax = fig.add_axes([.145, .585, .78, .235])
cols2 = ['#b8331f', '#d2601f', '#d6a520', '#5b9bd5', '#3f6fd1']
bb = ax.bar(lbls, rates, color=cols2, width=.6, alpha=.9)
for b, r in zip(bb, rates):
    ax.text(b.get_x() + b.get_width() / 2, r + .25, f'{r:.1f}%', ha='center',
            fontsize=11.5, weight='bold')
ax.set_ylim(0, max(rates) * 1.35)
ax.set_xlabel('금속함량 [Fe/H]   ← 금속이 적은 별', fontsize=11)
ax.set_ylabel('오분류율 (%)', fontsize=11)
ax.tick_params(axis='x', labelsize=9.5)
ax.grid(axis='y', alpha=.45); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ways = [('01', '백색왜성',
         'test 표본 43개 중 재현율 30.2%. 연속선 정규화가 백색왜성 특유의\n'
         '매우 넓은 수소선 날개를 연속선으로 착각해 깎아내기 때문이다.\n'
         '→ 정규화를 우회하는 별도 경로가 필요하다.'),
        ('02', '금속이 적은 별',
         f'금속 결핍 별의 오분류율 {rates[0]:.1f}% 로 금속이 풍부한 별({rates[-1]:.1f}%)의 5배 이상.\n'
         '흡수선이 얕아져 더 뜨거운 별처럼 보이는 온도-금속함량 축퇴 때문이다.\n'
         '→ 금속함량을 함께 예측하는 학습이 해법이다.'),
        ('03', 'O형 표본',
         'O형 별은 수 자체가 매우 드물어 단독 학습이 어려웠고 B형과 묶어\n'
         'OB 로 다루었다.  → 고온성 관측을 더 확보하면 분리해 낼 수 있다.')]
for i, (no, head, body) in enumerate(ways):
    yy = .525 - i * .105
    fig.text(L + .004, yy, no, fontsize=12, weight='bold', color=FAINT, va='top')
    fig.text(L + .042, yy, head, fontsize=13.5, weight='bold', va='top')
    fig.text(L + .042, yy - .031, body, fontsize=10.5, color=MID, va='top',
             linespacing=1.75)

fig.add_artist(Rectangle((L, .148), R - L, .055, transform=fig.transFigure,
                         facecolor='#0f1218', lw=0))
fig.text(L + .022, .1755, '프로그램과 코드를 공개했습니다', fontsize=12,
         color='#ffffff', weight='bold', va='center')
fig.text(R - .022, .1755, 'github.com/jigooborn/nobellsang-batnenyonego',
         fontsize=10.5, color='#6ea8ff', va='center', ha='right')

takeaway(fig, '한계를 아는 것이 다음 연구의 출발점이다.\n세 과제 모두 원인을 물리적으로 특정했다.', color=OK)
save(fig, '10_한계와나아갈방향.png')

print('\n완료 →', OUT)
