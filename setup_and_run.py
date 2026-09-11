# -*- coding: utf-8 -*-
"""
setup_and_run.py — 필요한 라이브러리를 확인·설치하고 프로그램을 실행한다.

시작하기.bat 이 이 파일을 부른다. 배치 파일에 한글을 많이 넣으면
윈도우 cmd 가 goto 로 점프할 때 바이트 위치를 잘못 잡아 줄이 깨지므로,
안내 문구와 설치 로직은 모두 여기(파이썬)에 둔다.

  python setup_and_run.py             설치 확인 후 프로그램 실행
  python setup_and_run.py --diagnose  설치 문제 진단만 (진단결과.txt 저장)
"""
import importlib
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDED = ['numpy', 'scipy', 'pandas', 'astropy', 'matplotlib', 'torch']
LIGHT = ['numpy', 'scipy', 'pandas', 'astropy', 'matplotlib']
VCREDIST = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
PY312 = 'https://www.python.org/downloads/release/python-3127/'

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass


def say(msg=''):
    print(msg, flush=True)


def check():
    """미설치/불러오기 실패 목록과 오류를 돌려준다."""
    bad = {}
    for name in NEEDED:
        try:
            importlib.import_module(name)
        except Exception as e:
            bad[name] = e
    return bad


def pip(*args, user=False):
    cmd = [sys.executable, '-m', 'pip', 'install',
           '--timeout', '180', '--retries', '5']
    if user:
        cmd.append('--user')
    cmd += list(args)
    say('  > ' + ' '.join(cmd[2:]))
    return subprocess.call(cmd)


def why(err):
    """오류 내용을 보고 원인과 해결책을 찾아 준다."""
    t = '{}: {}'.format(type(err).__name__, err)
    low = t.lower()
    if 'dll load failed' in low or 'winerror 126' in low or \
            'specified module could not be found' in low:
        return ('Visual C++ 재배포 패키지가 없어서입니다. '
                '파이토치가 윈도우에서 쓰는 DLL 이라 반드시 필요합니다.',
                ['아래 주소에서 내려받아 설치하세요 (1분):',
                 '    ' + VCREDIST,
                 '설치한 뒤 컴퓨터를 다시 시작하고 시작하기.bat 을 다시 실행하세요.'])
    if 'numpy' in low and ('abi' in low or 'compiled' in low
                           or 'binary incompatib' in low):
        return ('넘파이 버전이 파이토치와 맞지 않습니다.',
                ['명령 프롬프트에서 아래를 실행하세요:',
                 '    "{}" -m pip install "numpy<2.5"'.format(sys.executable)])
    if 'no module named' in low:
        return ('설치 자체가 되지 않았습니다.',
                ['인터넷 연결과 방화벽을 확인한 뒤 다시 실행해 주세요.',
                 '학교나 회사 와이파이라면 차단될 수 있습니다.'])
    if 'illegal instruction' in low or 'avx' in low:
        return ('이 CPU 가 파이토치 최신 버전을 지원하지 않습니다.',
                ['아래로 이전 버전을 설치해 보세요:',
                 '    "{}" -m pip install torch==2.5.1'.format(sys.executable)])
    return ('원인을 자동으로 찾지 못했습니다.',
            ['아래로 이전 버전을 설치해 보세요:',
             '    "{}" -m pip install torch==2.5.1'.format(sys.executable),
             '그래도 안 되면 오류기록.txt 파일을 보내 주세요.'])


def report(bad, path):
    """오류 상세를 파일로 남긴다."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('[설치 오류 기록]\n')
        f.write('파이썬: {}\n'.format(sys.version.replace('\n', ' ')))
        f.write('실행파일: {}\n'.format(sys.executable))
        f.write('폴더: {}\n\n'.format(HERE))
        for name in NEEDED:
            if name in bad:
                f.write('--- {} : 실패 ---\n'.format(name))
                f.write(''.join(traceback.format_exception(
                    type(bad[name]), bad[name],
                    bad[name].__traceback__)) + '\n')
            else:
                try:
                    mod = importlib.import_module(name)
                    f.write('--- {} : OK {}\n'.format(
                        name, getattr(mod, '__version__', '')))
                except Exception:
                    pass


def diagnose():
    say('=' * 52)
    say(' 설치 진단')
    say('=' * 52)
    say('파이썬 : {}'.format(sys.version.split()[0]))
    say('실행파일: {}'.format(sys.executable))
    say('비트   : {}'.format('64bit' if sys.maxsize > 2 ** 32 else '32bit'))
    say('폴더   : {}'.format(HERE))
    say('최신 폴더 여부: {}'.format(
        '예' if os.path.exists(os.path.join(HERE, 'setup_and_run.py'))
        else '아니오 — 새로 내려받으세요'))
    say()
    bad = check()
    for name in NEEDED:
        if name in bad:
            say('  {:12s} 실패 → {}: {}'.format(
                name, type(bad[name]).__name__, str(bad[name])[:160]))
        else:
            mod = importlib.import_module(name)
            say('  {:12s} OK  {}'.format(
                name, getattr(mod, '__version__', '')))
    say()
    try:
        import urllib.request
        with urllib.request.urlopen('https://pypi.org/simple/torch/',
                                    timeout=20) as r:
            say('PyPI 연결: 정상 (응답 {})'.format(r.status))
    except Exception as e:
        say('PyPI 연결: 실패 → {}'.format(type(e).__name__))
    try:
        import shutil
        free = shutil.disk_usage(HERE).free / (1024 ** 3)
        say('디스크 여유: {:.1f} GB'.format(free))
    except Exception:
        pass
    say()
    if bad:
        cause, steps = why(list(bad.values())[0])
        say('원인: ' + cause)
        for s in steps:
            say('  ' + s)
        report(bad, os.path.join(HERE, '진단결과.txt'))
        say()
        say('자세한 내용을 진단결과.txt 로 저장했습니다.')
    else:
        say('모든 라이브러리가 정상입니다. 시작하기.bat 을 실행하세요.')
    return 0


def main():
    if '--diagnose' in sys.argv:
        return diagnose()

    say()
    say(' ' + '=' * 50)
    say('   항성 분광형 AI 자동 분류 프로그램')
    say('   목천고등학교  이한결 · 맹진호')
    say(' ' + '=' * 50)
    say()
    say(' [확인] 파이썬 {}'.format(sys.version.split()[0]))
    say()

    bad = check()
    if bad:
        say(' [1/2] 필요한 라이브러리를 설치합니다.')
        say('       처음 한 번만 5~10분 걸립니다. 창을 닫지 마세요.')
        say()
        subprocess.call([sys.executable, '-m', 'pip', 'install',
                         '--upgrade', 'pip'])
        light = [n for n in LIGHT if n in bad]
        if light:
            pip(*light)
        if 'torch' in bad:
            say()
            say(' [설치] 파이토치는 용량이 커서 시간이 걸립니다. (약 130MB)')
            say()
            pip('torch')
        bad = check()

    if bad:                                   # 사용자 폴더로 재시도
        say()
        say(' [재시도] 사용자 폴더에 설치합니다...')
        say()
        pip(*[n for n in NEEDED if n in bad], user=True)
        bad = check()

    if bad:
        log = os.path.join(HERE, '오류기록.txt')
        report(bad, log)
        cause, steps = why(list(bad.values())[0])
        say()
        say(' ' + '=' * 50)
        say('  [!] 준비가 끝나지 않았습니다.')
        say(' ' + '=' * 50)
        say()
        for name in NEEDED:
            mark = '실패' if name in bad else 'OK'
            say('   {:12s} {}'.format(name, mark))
        say()
        say(' 오류 내용:')
        for name, e in bad.items():
            say('   [{}] {}: {}'.format(name, type(e).__name__,
                                        str(e)[:200]))
        say()
        say(' 원인: ' + cause)
        say()
        for s in steps:
            say('   ' + s)
        say()
        say(' 자세한 내용을 오류기록.txt 로 저장했습니다.')
        say(' 해결이 안 되면 그 파일을 보내 주세요.')
        say()
        return 1

    say(' [2/2] 프로그램을 실행합니다.')
    say()
    return subprocess.call([sys.executable,
                            os.path.join(HERE, 'classify_gui_v5.py')])


if __name__ == '__main__':
    sys.exit(main())
