# -*- coding: utf-8 -*-
"""
证据归档打包（只读，不训练、不改权重）
=====================================
用途：把服务器上散落的实验证据一次性打包回传，用于
  1) 清单待办「开源仓库：实验脚本 + 划分文件 + 全部 212 run 的 final_metrics」
  2) 论文附录 / 给导师的证据包
  3) 两次 TFA 自查的 json + 叠放图归档

不打包 .pt 权重（426 个 × 21.5 MiB ≈ 9 GB，太大）；只收 json / yaml / py / png / md。
复用 tfa_selfcheck.py 的 find_root（同目录）。

用法（服务器 /root 下）：
    python /root/archive_evidence.py

输出：
    /root/evidence_pack.zip         <- 回传这个
    /root/evidence_manifest.json    <- 内容清单（也在 zip 内）
zip 内目录：
    final_metrics/   全部 run 的 final_metrics.json（按相对路径重命名，避免同名覆盖）
    selfcheck/       两次自查的 json
    vis/             两次自查的叠放图
    data_yaml/       划分 yaml
    scripts/         src/*.py 与顶层 .py
    manifest.json    清单与计数
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import tfa_selfcheck as T
except Exception as e:  # noqa
    print('[FATAL] 无法 import tfa_selfcheck: %s' % e)
    print('        请把 tfa_selfcheck.py 与 tfa_stage2.py 一起放在 /root/')
    sys.exit(2)

EXPECTED_RUNS = 212
SKIP_SUFFIX = {'.pt', '.pth', '.pkl', '.pyc', '.onnx', '.engine', '.npy', '.npz'}
SKIP_DIR = {'__pycache__', '.git', '.cache', 'runs'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/evidence_pack.zip',
                    help='zip 输出路径（默认 /root/evidence_pack.zip）')
    ap.add_argument('--base', default='auto',
                    help='找 ROOT 的起点；auto = 先 /root/autodl-tmp 再 /root')
    a = ap.parse_args()

    OUT = Path(a.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mp = OUT.parent / (OUT.stem + '_manifest.json')

    bases = (['/root/autodl-tmp', '/root'] if a.base == 'auto' else [a.base])
    root, hits = None, []
    for b in bases:
        bp = Path(b)
        if not bp.is_dir():
            continue
        print('[SCAN] %s ...' % b, flush=True)
        root, hits = T.find_root(bp)
        if root is not None:
            break
    if root is None:
        print('[FATAL] 找不到 ROOT（需含 src/train_mvp.py）')
        sys.exit(3)
    print('[OK] ROOT = %s' % root, flush=True)

    groups = {}

    # 1) 全部 final_metrics.json
    fm = sorted(root.rglob('final_metrics.json'))
    groups['final_metrics'] = fm
    print('[OK] final_metrics.json = %d （目标 %d）'
          % (len(fm), EXPECTED_RUNS), flush=True)

    # 2) 两次自查 json
    names = ['tfa_scan_report.json', 'tfa_selfcheck_result.json',
             'tfa_stage2_result.json']
    sc = []
    for n in names:
        p = root / '输出' / n
        if p.is_file():
            sc.append(p)
    groups['selfcheck'] = sc
    print('[OK] 自查 json = %d' % len(sc), flush=True)

    # 3) 叠放图
    vis = []
    for d in ['tfa_selfcheck_vis', 'tfa_stage2_vis']:
        v = root / '输出' / d
        if v.is_dir():
            vis += sorted(v.rglob('*.png'))
    groups['vis'] = vis
    print('[OK] 叠放图 = %d' % len(vis), flush=True)

    # 4) 划分 yaml（只取 mvp_data，排除 neu）
    yd = root / '输出' / 'mvp_data'
    yamls = sorted(yd.glob('*.yaml')) if yd.is_dir() else []
    groups['data_yaml'] = yamls
    print('[OK] 划分 yaml = %d' % len(yamls), flush=True)

    # 5) 脚本
    scripts = []
    for pat in ['src/*.py', '*.py']:
        for p in sorted(root.glob(pat)):
            if p.suffix.lower() in SKIP_SUFFIX:
                continue
            if any(part in SKIP_DIR for part in p.parts):
                continue
            if p not in scripts:
                scripts.append(p)
    groups['scripts'] = scripts
    print('[OK] 脚本 = %d' % len(scripts), flush=True)

    # 顶层说明文档
    docs = [p for p in sorted(root.glob('*.md'))]
    groups['docs'] = docs
    print('[OK] 顶层 md = %d' % len(docs), flush=True)

    try:
        import ultralytics
        ver = getattr(ultralytics, '__version__', 'unknown')
    except Exception:
        ver = 'not-imported'

    manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'root': str(root),
        'ultralytics': ver,
        'expected_final_metrics': EXPECTED_RUNS,
        'counts': {k: len(v) for k, v in groups.items()},
        'notes': [
            '权重 .pt 未打包（体积过大）；如需权重请单独回传。',
            'final_metrics 按原相对路径重命名后存入 final_metrics/，避免同名覆盖。',
            '两次自查结论见 tfa自查判定_v1.md（本地），本包只含一手数据。',
        ],
    }
    if len(fm) < EXPECTED_RUNS:
        manifest['warnings'] = [
            'final_metrics 实际 %d 份，少于目标 %d 份，缺 %d 份'
            % (len(fm), EXPECTED_RUNS, EXPECTED_RUNS - len(fm))
        ]
        print('[WARN] final_metrics 缺 %d 份，已在 manifest 标注'
              % (EXPECTED_RUNS - len(fm)), flush=True)

    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    prefix = {
        'final_metrics': 'final_metrics',
        'selfcheck': 'selfcheck',
        'vis': 'vis',
        'data_yaml': 'data_yaml',
        'scripts': 'scripts',
        'docs': 'docs',
    }

    def arcname(p, key):
        rel = p.relative_to(root).as_posix()
        if key == 'final_metrics':
            # 用目录层级拼唯一名，避免 212 份同名文件互相覆盖
            safe = rel.replace('/', '__')
            return 'final_metrics/' + safe
        if key == 'vis':
            # 保留 split 子目录
            try:
                sub = p.relative_to(root / '输出').as_posix()
            except ValueError:
                sub = rel
            return 'vis/' + sub
        return prefix[key] + '/' + p.name

    total = 0
    with zipfile.ZipFile(str(OUT), 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(str(mp), 'manifest.json')
        total += 1
        for key, items in groups.items():
            for p in items:
                z.write(str(p), arcname(p, key))
                total += 1
    print('\n[OK] zip -> %s  (%.1f KB, %d 个条目)'
          % (OUT, OUT.stat().st_size / 1024, total), flush=True)
    print('[OK] manifest -> %s' % mp, flush=True)
    print('[NEXT] FileZilla 回传：%s' % OUT, flush=True)
    for k, v in manifest['counts'].items():
        print('       %-16s %d' % (k, v), flush=True)


if __name__ == '__main__':
    main()
