# -*- coding: utf-8 -*-
"""
TFA 三项自查 · 加固版（Stage 2）
================================
背景：tfa_selfcheck.py v2 只在 shot100_s1 采集了 ②框数/置信度、③IoU 分档 + 可视化
      （源码 `if split == splits[0]`），mAP50-95 逐类才是 s1..s5 全覆盖。
本脚本把 ②③ 扩到 s1..s5 全部 5 个抽样，使「排除框数/置信度伪影」从单点变 5 配对。

不训练，只 val + predict。复用 tfa_selfcheck.py 里已验证的函数（同目录）。
依赖：ultralytics, torch, opencv-python, numpy（与 v2 相同）

用法（服务器 /root 下）：
    python /root/tfa_stage2.py              # 全量 5 split + 打包
    python /root/tfa_stage2.py --dry-run    # 只盘点权重，不推理（30 秒）
    python /root/tfa_stage2.py --vis-per-split 2

输出：
    ROOT/输出/tfa_stage2_result.json
    ROOT/输出/tfa_stage2_vis/<split>/*.png
    ROOT/tfa_stage2.zip          <- 回传这个
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import tfa_selfcheck as T
except Exception as e:  # noqa
    print('[FATAL] 无法 import tfa_selfcheck: %s' % e)
    print('        请确认 tfa_selfcheck.py 与本脚本在同一个目录（默认 /root）')
    sys.exit(2)

SPLITS = T.SPLITS
GROUPS = T.GROUPS


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def iou_dist_chunked(model, imgs, label_index, label, chunk=100):
    """分块调用 T.iou_distribution，合并 bins，并打印进度（避免看起来卡死）。"""
    merged = {'n_boxes': 0, 'bins': {}}
    n = len(imgs)
    for i in range(0, n, chunk):
        part = imgs[i:i + chunk]
        r = T.iou_distribution(model, part, label_index, label)
        if 'error' in r:
            r['partial_n_boxes'] = merged['n_boxes']
            return r
        merged['n_boxes'] += int(r.get('n_boxes', 0))
        for k, v in (r.get('bins') or {}).items():
            merged['bins'][k] = merged['bins'].get(k, 0) + int(v)
        print('      iou %d/%d' % (min(i + chunk, n), n), flush=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='auto',
                    help="从哪个目录找 ROOT；auto = 先 /root/autodl-tmp 再 /root")
    ap.add_argument('--dry-run', action='store_true', help='只盘点权重，不推理')
    ap.add_argument('--vis-per-split', type=int, default=2,
                    help='每个 split 生成几张叠放对比图（默认 2）')
    a = ap.parse_args()

    if a.base == 'auto':
        bases = ['/root/autodl-tmp', '/root']
    else:
        bases = [a.base]

    root, hits = None, []
    for b in bases:
        bp = Path(b)
        if not bp.is_dir():
            print('[SKIP] %s 不存在' % b, flush=True)
            continue
        print('[SCAN] 正在扫描 %s ...' % b, flush=True)
        root, hits = T.find_root(bp)
        if root is not None:
            break
    if root is None:
        print('[FATAL] find_root 失败：在 %s 下没找到 src/train_mvp.py' % bases)
        print('        可用 --base 指定上一级目录，例如 --base /root/autodl-tmp')
        sys.exit(3)
    print('[OK] ROOT = %s' % root)
    print('[OK] train_mvp.py 命中 %d 处' % len(hits))

    pts = T.all_pt_files(root)
    print('[OK] .pt 权重数 = %d' % len(pts))
    if not pts:
        print('[FATAL] 全 ROOT 无 .pt：走审读意见 §5.3 降级（数据不删、解释降级、摘要删去），不重训')
        sys.exit(4)

    ydir = root / '输出' / 'mvp_data'
    by_split = {s: ydir / (s + '.yaml') for s in SPLITS}
    ok_yaml = [s for s in SPLITS if by_split[s].is_file()]
    print('[OK] 可用 yaml = %s' % ok_yaml)
    if not ok_yaml:
        print('[FATAL] %s 下无 shot100_s1..s5.yaml' % ydir)
        sys.exit(5)

    if a.dry_run:
        for s in ok_yaml:
            for g in GROUPS:
                pt, rn = T.find_weight(root, g, s, pts)
                print('   %-12s %-10s -> %-70s | run_name=%s'
                      % (s, g, (str(pt) if pt else 'MISSING'), rn))
        print('[DRY-RUN] 结束，未跑推理')
        return

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa
        print('[FATAL] ultralytics 导入失败: %s' % e)
        sys.exit(6)

    result = {'splits': {}, 'notes': [], 'val_images': None, 'label_count': None,
              'names': {}, 'per_split_ok': {}}
    vis_root = root / '输出' / 'tfa_stage2_vis'
    jp = root / '输出' / 'tfa_stage2_result.json'
    t0 = time.time()

    for si, split in enumerate(ok_yaml, 1):
        print('\n===== [%d/%d] %s =====' % (si, len(ok_yaml), split), flush=True)
        entry, pair = {}, {}

        for g in GROUPS:
            pt, rn = T.find_weight(root, g, split, pts)
            if pt is None:
                entry[g] = {'missing': rn}
                print('  [MISS] %s -> %s' % (g, rn), flush=True)
                continue
            entry[g] = {'weight': str(pt.relative_to(root)), 'run_name': rn,
                        'size_MiB': round(pt.stat().st_size / 1048576, 1)}
            try:
                pair[g] = YOLO(str(pt))
                print('  [LOAD] %s -> %s' % (g, pt.name), flush=True)
            except Exception as e:  # noqa
                entry[g]['load_error'] = '%s: %s' % (type(e).__name__, e)

        d, _b, img_dir, imgs, label_index = T.resolve_data(root, by_split[split])
        if result['val_images'] is None:
            result['val_images'] = len(imgs)
            result['label_count'] = len(label_index)
            result['names'] = d.get('names', {})
        print('  val 图片 = %d  标签 = %d  img_dir = %s'
              % (len(imgs), len(label_index), img_dir), flush=True)

        for g, model in pair.items():
            # ① val：mAP50 / mAP50-95 / mAP75 / 逐类
            try:
                m = model.val(data=str(by_split[split]), split='val', conf=0.001,
                              save_json=False, plots=False, verbose=False)
                bb = m.box
                entry[g]['val'] = {
                    'map50': float(bb.map50),
                    'map50_95': float(bb.map),
                    'map75': float(bb.map75) if hasattr(bb, 'map75') else None,
                    'maps_per_class_mAP50_95': [float(x) for x in bb.maps],
                    'ap50_per_class': ([float(x) for x in bb.ap50]
                                       if hasattr(bb, 'ap50') else None),
                    'ap_class_index': ([int(x) for x in bb.ap_class_index]
                                       if hasattr(bb, 'ap_class_index') else None),
                }
                print('  [VAL] %s  map50=%.4f  map50-95=%.4f'
                      % (g, bb.map50, bb.map), flush=True)
            except Exception as e:  # noqa
                entry[g]['val_error'] = '%s: %s' % (type(e).__name__, e)
                print('  [VAL-FAIL] %s %s' % (g, str(e)[:120]), flush=True)

            # ② 框数 / 置信度分布（每个 split 都做）
            entry[g]['box_conf'] = T.box_conf_stats(model, imgs, g)
            print('  [BOX] %s  boxes=%s' % (g, entry[g]['box_conf'].get('total_boxes')),
                  flush=True)

            # ③ IoU 分档（每个 split 都做，分块带进度）
            if label_index:
                entry[g]['iou_dist'] = iou_dist_chunked(model, imgs, label_index, g)
            else:
                entry[g]['iou_dist'] = {'skip': 'no label txt'}
            print('  [IOU] %s  n_boxes=%s' % (g, entry[g]['iou_dist'].get('n_boxes')),
                  flush=True)

        # ③ 同图叠放（每个 split 取不同图片段，覆盖更多样本）
        if len(pair) == 2 and imgs:
            k = max(1, a.vis_per_split)
            start = (si - 1) * k
            seg = imgs[start:start + k] or imgs[:k]
            entry['visual'] = T.visual_compare(
                [pair[g] for g in GROUPS], seg, vis_root / split,
                list(GROUPS), max_imgs=k)

        complete = all(
            ('val' in entry.get(g, {}))
            and ('box_conf' in entry.get(g, {}))
            and ('iou_dist' in entry.get(g, {}))
            for g in GROUPS
        )
        entry['complete'] = complete
        result['per_split_ok'][split] = complete
        if not complete:
            result['notes'].append('%s: 字段不全' % split)
        result['splits'][split] = entry

        jp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        pair.clear()
        _empty_cache()
        print('  [DONE] %s complete=%s elapsed=%.1f min'
              % (split, complete, (time.time() - t0) / 60), flush=True)

    print('\n[OK] result -> %s' % jp)
    nok = sum(1 for v in result['per_split_ok'].values() if v)
    print('[OK] 完整 split = %d/%d' % (nok, len(result['per_split_ok'])))

    zp = root / 'tfa_stage2.zip'
    files = [jp]
    if vis_root.is_dir():
        files += sorted(vis_root.rglob('*.png'))
    with zipfile.ZipFile(str(zp), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(str(f), str(f.relative_to(root)))
    print('[OK] zip -> %s  (%.1f KB, %d 个文件)'
          % (zp, zp.stat().st_size / 1024, len(files)))
    print('[NEXT] 用 FileZilla 回传：%s' % zp)


if __name__ == '__main__':
    main()
