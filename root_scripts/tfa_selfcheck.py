#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tfa_selfcheck.py  v2 (2026-11-21)

P1-7 (--freeze 10) 的 +9.12 mAP50-95 反转三项自查数据采集。

v1 回传的 zip 是空壳（5 个划分全部 missing），定位到 3 个脚本 bug，v2 全部修复：
  1) 权重不再假定「输出/runs_tfa 下叫 last.pt」。train_mvp.py 用的是
     model.trainer.save_dir，Ultralytics 可能把 project 嵌套进别的目录。
     改为全 ROOT 扫描 *.pt，用 final_metrics.json 的 config.run_name 做
     路径组件精确匹配（parts 匹配，天然排除 s1/s10 前缀混淆）。
  2) 划分显式锁 shot100_s1..s5，yaml 只取 输出/mvp_data（排除 mvp_data_neu），
     避免字符串排序把 shot100_s10 排进前 5、漏掉 s5。
  3) val 路径按 yaml 的 path(绝对) + val 组合解析；names 支持多行字典；
     标签用 path 下的 stem->txt 索引，不再硬拼 images->labels。

若全 ROOT 无任何 .pt（权重已被清理），--export 不空跑，明确提示走
《论文骨架审读意见》§5.3 降级（数据不删、解释降级、摘要删去，不重训），
并仍打包诊断报告供回传定案。

用法（脚本放哪都行，自己定位 ROOT）：
  python tfa_selfcheck.py            # Stage 1 只读盘点 -> 打印 + tfa_scan_report.json
  python tfa_selfcheck.py --export   # Stage 1 + Stage 2 -> tfa_selfcheck.zip

Stage 2 只做 val / predict，不训练、不修改任何权重。
"""

import os
import re
import sys
import json
import zipfile
import pathlib
import argparse
import traceback
import collections

BASE = pathlib.Path('/root/autodl-tmp')
IOU_BINS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = ['shot100_s%d' % i for i in range(1, 6)]
GROUPS = ('runs_tfa', 'runs_b8a')


def log(*a):
    print(*a, flush=True)


def find_root(base):
    hits = [p for p in base.rglob('train_mvp.py') if p.parent.name == 'src']
    if not hits:
        return None, []
    pref = [p for p in hits if (p.parent.parent / '输出' / 'runs_tfa').is_dir()]
    return (pref or hits)[0].parent.parent, hits


def all_pt_files(root):
    return sorted(root.rglob('*.pt'))


def parse_yaml(path):
    """不依赖 yaml 库：抓 path/val/train/nc + names（行内列表或多行字典）。"""
    out = {'names': {}, 'path': None, 'val': None, 'train': None, 'nc': None}
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        return out
    in_names = False
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if s.lower().startswith('names:'):
            rest = s.split(':', 1)[1].strip()
            in_names = True
            if rest.startswith('[') and rest.endswith(']'):
                for i, nm in enumerate(rest[1:-1].split(',')):
                    out['names'][i] = nm.strip().strip("'").strip('"')
            elif ':' in rest:
                ci, nm = rest.split(':', 1)
                try:
                    out['names'][int(ci.strip())] = nm.strip().strip("'").strip('"')
                except ValueError:
                    pass
            continue
        if ':' not in s:
            continue
        k, v = s.split(':', 1)
        k = k.strip().lower()
        v = v.strip().strip("'").strip('"')
        if in_names and re.match(r'^\d+\s*:', s):
            ci, nm = s.split(':', 1)
            try:
                out['names'][int(ci.strip())] = nm.strip().strip("'").strip('"')
            except ValueError:
                pass
            continue
        if k in ('path', 'val', 'train', 'nc'):
            out[k] = v
            in_names = False
    return out


def resolve_data(root, yaml_path):
    """按 yaml 的 path + val 组合定位图片，并在 path 下建 stem->txt 标签索引。"""
    d = parse_yaml(yaml_path)
    p = d.get('path')
    if p and pathlib.Path(p).is_absolute():
        base = pathlib.Path(p)
    elif p:
        base = yaml_path.parent / p
    else:
        base = yaml_path.parent
    vr = d.get('val') or ''
    img_dir = None
    if vr:
        c = pathlib.Path(vr)
        img_dir = c if c.is_absolute() else (base / c)
    if img_dir and img_dir.is_file():
        img_dir = img_dir.parent
    imgs = []
    if img_dir and img_dir.is_dir():
        imgs = sorted(q for q in img_dir.rglob('*') if q.suffix.lower() in IMG_EXT)
    label_index = {}
    if base.is_dir():
        for t in base.rglob('*.txt'):
            label_index.setdefault(t.stem, t)
    return d, base, img_dir, imgs, label_index


def stage1(root):
    rep = {'root': str(root), 'all_pt': {}, 'groups': {}, 'yaml': {},
           'ultralytics': None, 'errors': []}
    pts = all_pt_files(root)
    rep['all_pt']['count'] = len(pts)
    rep['all_pt']['paths'] = [str(p.relative_to(root)) for p in pts[:80]]
    rep['all_pt']['truncated'] = len(pts) > 80

    for g in GROUPS:
        proj = root / '输出' / g
        e = {'exists': proj.is_dir()}
        if proj.is_dir():
            fs = sorted(q for q in proj.rglob('*') if q.is_file())
            e['n_files'] = len(fs)
            e['file_sample'] = [str(q.relative_to(root)) for q in fs[:25]]
            e['first_level_dirs'] = sorted(q.name for q in proj.iterdir() if q.is_dir())
            metrics = sorted(proj.glob('*/final_metrics.json'))
            e['n_final_metrics'] = len(metrics)
            if metrics:
                try:
                    mm = json.loads(metrics[0].read_text(encoding='utf-8'))
                    e['run_name_sample'] = mm.get('config', {}).get('run_name')
                    e['ckpt_sample'] = mm.get('config', {}).get('ckpt')
                    e['split_sample'] = mm.get('split')
                except Exception as ex:
                    e['parse_error'] = str(ex)
        rep['groups'][g] = e

    ydir = root / '输出' / 'mvp_data'
    ys = [ydir / (s + '.yaml') for s in SPLITS]
    rep['yaml']['dir'] = str(ydir)
    rep['yaml']['expected'] = [y.name for y in ys]
    rep['yaml']['found'] = [y.name for y in ys if y.is_file()]
    infos = {}
    for y in ys:
        if not y.is_file():
            continue
        d, base, img_dir, imgs, lidx = resolve_data(root, y)
        infos[y.stem] = {
            'path': d.get('path'), 'val': d.get('val'), 'train': d.get('train'),
            'names': d.get('names'), 'img_dir': str(img_dir) if img_dir else None,
            'img_dir_exists': bool(img_dir and img_dir.is_dir()),
            'n_images': len(imgs), 'n_label_txt': len(lidx),
        }
    rep['yaml']['parsed'] = infos
    rep['yaml']['all_shot100_yaml'] = [
        str(q.relative_to(root)) for q in sorted(root.rglob('shot100_s*.yaml'))[:25]]

    try:
        import importlib.metadata as M
        rep['ultralytics'] = M.version('ultralytics')
    except Exception as e:
        rep['errors'].append('ultralytics 版本查询失败: %s' % e)
    return rep


def print_report(rep):
    log('ROOT =', rep['root'])
    log('ultralytics =', rep['ultralytics'])
    log('=== 1) 全 ROOT .pt 权重 ===', rep['all_pt']['count'], '个')
    for x in rep['all_pt']['paths'][:20]:
        log('   ', x)
    if rep['all_pt']['truncated']:
        log('    ...(清单截断，完整版见 tfa_scan_report.json)')
    for g in GROUPS:
        e = rep['groups'][g]
        log('=== 2) %s ===' % g, '存在' if e.get('exists') else '不存在',
            '| 文件数', e.get('n_files'), '| final_metrics', e.get('n_final_metrics'),
            '| run_name 样例', e.get('run_name_sample'))
        for x in e.get('file_sample', [])[:8]:
            log('   ', x)
    log('=== 3) shot100 yaml（只取 输出/mvp_data，显式 s1..s5）===')
    log('   期望:', rep['yaml']['expected'])
    log('   找到:', rep['yaml']['found'])
    for k, v in rep['yaml']['parsed'].items():
        log('   %s: img_dir=%s exists=%s images=%d labels=%d' % (
            k, v['img_dir'], v['img_dir_exists'], v['n_images'], v['n_label_txt']))
        log('      names=%s' % v['names'])
    if rep['errors']:
        log('=== 提示 ===')
        for e in rep['errors']:
            log('   ', e)
    if rep['all_pt']['count'] == 0:
        log('!!! 全 ROOT 无 .pt 权重 -> 三项自查（重跑 val）不可行。')
        log('    走审读意见 §5.3：数据不删、解释降级为「未解释现象」、摘要删去；不要重训。')
    log('=== Stage 1 结束（全程只读）===')


def find_weight(root, group, split, pt_index):
    """用 final_metrics.json 的 config.run_name 做路径组件精确匹配。"""
    proj = root / '输出' / group
    run_name = None
    for f in sorted(proj.glob('*/final_metrics.json')):
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        cfg = d.get('config', {})
        rn = cfg.get('run_name') or ''
        same = (d.get('split') == split) or bool(
            re.search(re.escape(split) + r'(?!\d)', rn))
        if same:
            run_name = rn or None
            break
    if not run_name:
        return None, 'no final_metrics.json matched %s' % split
    cands = [p for p in pt_index if run_name in p.parts]
    c2 = [p for p in cands if group in p.parts]
    if c2:
        cands = c2
    if not cands:
        cands = [p for p in pt_index
                 if any(part == split or part.startswith(split + '_') for part in p.parts)
                 and group in str(p)]
    cands.sort(key=lambda p: (0 if p.name == 'last.pt'
                              else 1 if p.name == 'best.pt' else 2, len(str(p))))
    return (cands[0] if cands else None), run_name


def box_conf_stats(model, images, label, conf=0.001):
    hist = collections.Counter()
    per_cls = collections.Counter()
    total = 0
    confs = []
    try:
        for r in model.predict(images, conf=conf, verbose=False, stream=True):
            b = r.boxes
            if b is None:
                continue
            for c, s in zip(b.cls.tolist(), b.conf.tolist()):
                total += 1
                confs.append(float(s))
                per_cls[int(c)] += 1
                hist[min(9, int(float(s) * 10))] += 1
    except Exception as e:
        return {'error': '%s: %s' % (type(e).__name__, e),
                'trace': traceback.format_exc(limit=2)}
    out = {'total_boxes': total, 'per_class': dict(per_cls),
           'conf_hist_10bins': dict(hist)}
    if confs:
        confs.sort()
        out['conf_min'] = confs[0]
        out['conf_median'] = confs[len(confs) // 2]
        out['conf_mean'] = sum(confs) / len(confs)
        out['conf_max'] = confs[-1]
        out['boxes_above_0p25'] = sum(1 for x in confs if x >= 0.25)
        out['boxes_above_0p5'] = sum(1 for x in confs if x >= 0.5)
    return out


def iou_of(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_boxes(label_path, wh):
    out = []
    try:
        W, H = wh
        for line in label_path.read_text(encoding='utf-8', errors='replace').splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            out.append((c, (x1, y1, x2, y2)))
    except Exception:
        pass
    return out


def iou_distribution(model, img_paths, label_index, label):
    bins = collections.Counter()
    n_boxes = 0
    try:
        import cv2
        for img in img_paths:
            lp = label_index.get(img.stem)
            if lp is None:
                continue
            im = cv2.imread(str(img))
            if im is None:
                continue
            H, W = im.shape[:2]
            gts = gt_boxes(lp, (W, H))
            if not gts:
                continue
            rs = model.predict(str(img), conf=0.001, verbose=False)
            if not rs or rs[0].boxes is None:
                continue
            for c, xyxy in zip(rs[0].boxes.cls.tolist(),
                               rs[0].boxes.xyxy.tolist()):
                c = int(c)
                same = [g[1] for g in gts if g[0] == c]
                if not same:
                    bins['no_same_class_gt'] += 1
                    n_boxes += 1
                    continue
                m = max(iou_of(xyxy, g) for g in same)
                b = '<0.50'
                for t in IOU_BINS:
                    if m >= t:
                        b = '>=%.2f' % t
                bins[b] += 1
                n_boxes += 1
    except Exception as e:
        return {'error': '%s: %s' % (type(e).__name__, e),
                'trace': traceback.format_exc(limit=2), 'n_boxes': n_boxes,
                'bins': dict(bins)}
    return {'n_boxes': n_boxes, 'bins': dict(bins)}


def visual_compare(models, img_paths, out_dir, model_names, max_imgs=6):
    try:
        import cv2
    except Exception as e:
        return {'error': 'cv2 不可用: %s' % e}
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [(0, 0, 255), (0, 255, 0)]
    made = []
    for img in img_paths[:max_imgs]:
        try:
            im = cv2.imread(str(img))
            if im is None:
                continue
            canvas = im.copy()
            for mi, m in enumerate(models):
                rs = m.predict(str(img), conf=0.25, verbose=False)
                if not rs or rs[0].boxes is None:
                    continue
                for xyxy in rs[0].boxes.xyxy.tolist():
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[mi % 2], 2)
                    cv2.putText(canvas, model_names[mi][:2], (x1, max(12, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[mi % 2], 1)
            dst = out_dir / (img.stem + '_cmp.png')
            cv2.imwrite(str(dst), canvas)
            made.append(dst.name)
        except Exception as e:
            log('   [可视对照跳过]', img.name, type(e).__name__, str(e)[:80])
    return {'files': made, 'legend': {'red': model_names[0], 'green': model_names[1]}}


def run_export(root, rep):
    os.chdir(str(root))
    result = {'splits': {}, 'notes': []}
    artifacts = []
    try:
        from ultralytics import YOLO
    except Exception as e:
        return {'fatal': 'ultralytics 导入失败: %s' % e}, artifacts
    pt_index = all_pt_files(root)
    if not pt_index:
        return {'fatal': '全 ROOT 无 .pt 权重：三项自查不可行，走审读意见 §5.3 降级（不重训）'}, artifacts

    ydir = root / '输出' / 'mvp_data'
    by_split = {s: ydir / (s + '.yaml') for s in SPLITS}
    missing_yaml = [s for s, p in by_split.items() if not p.is_file()]
    if missing_yaml:
        result['notes'].append('缺 yaml: %s' % missing_yaml)
    splits = [s for s in SPLITS if by_split[s].is_file()]
    log('将处理划分（显式 s1..s5）:', splits)

    if not splits:
        return {'fatal': '输出/mvp_data 下无 shot100_s1..s5.yaml'}, artifacts
    d0, base, img_dir, all_imgs, label_index = resolve_data(root, by_split[splits[0]])
    log('val 图片数:', len(all_imgs), ' 标签索引:', len(label_index))
    result['val_images'] = len(all_imgs)
    result['label_count'] = len(label_index)
    result['names'] = d0.get('names', {})

    for split in splits:
        entry = {}
        pair = {}
        for group in GROUPS:
            pt, rn = find_weight(root, group, split, pt_index)
            if pt is None:
                entry[group] = {'missing': rn, 'pt_index_size': len(pt_index)}
                continue
            entry[group] = {'weight': str(pt.relative_to(root)), 'run_name': rn,
                            'size_MiB': round(pt.stat().st_size / 1048576, 1)}
            try:
                pair[group] = YOLO(str(pt))
            except Exception as e:
                entry[group]['load_error'] = '%s: %s' % (type(e).__name__, e)

        for group, model in pair.items():
            try:
                m = model.val(data=str(by_split[split]), split='val',
                              conf=0.001, save_json=False, plots=False,
                              verbose=False)
                b = m.box
                entry[group]['val'] = {
                    'map50': float(b.map50), 'map50_95': float(b.map),
                    'map75': float(b.map75) if hasattr(b, 'map75') else None,
                    'maps_per_class_mAP50_95': [float(x) for x in b.maps],
                    'ap50_per_class': [float(x) for x in b.ap50] if hasattr(b, 'ap50') else None,
                    'ap_class_index': [int(x) for x in b.ap_class_index],
                    'names': dict(m.names),
                }
                log('   val', split, group,
                    'map50=%.4f map50-95=%.4f' % (b.map50, b.map))
            except Exception as e:
                entry[group]['val_error'] = '%s: %s' % (type(e).__name__, e)
                entry[group]['val_trace'] = traceback.format_exc(limit=2)
                log('   [val 失败]', group, split, str(e)[:100])

        wa = entry.get('runs_tfa', {}).get('weight')
        wb = entry.get('runs_b8a', {}).get('weight')
        if wa and wb and wa == wb:
            result['notes'].append('%s: 两组匹配到同一权重，需人工核对目录' % split)

        if split == splits[0] and len(pair) == 2 and all_imgs:
            for group, model in pair.items():
                entry[group]['box_conf'] = box_conf_stats(model, all_imgs, group)
                entry[group]['iou_dist'] = (iou_distribution(
                    model, all_imgs, label_index, group)
                    if label_index else {'skip': 'no label txt'})
            entry['visual'] = visual_compare(
                [pair[g] for g in GROUPS], all_imgs,
                root / '输出' / 'tfa_selfcheck_vis', list(GROUPS))
        result['splits'][split] = entry

    jp = root / '输出' / 'tfa_selfcheck_result.json'
    jp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    artifacts.append(jp)
    vis = root / '输出' / 'tfa_selfcheck_vis'
    if vis.is_dir():
        artifacts += sorted(vis.glob('*.png'))
    return result, artifacts


def pack(root, rp, files):
    zp = root / '输出' / 'tfa_selfcheck.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in [rp] + list(files):
            if f.is_file():
                z.write(f, str(f.relative_to(root)))
    log('打包:', zp, zp.stat().st_size, 'bytes')
    log('内含:', [str(f.relative_to(root)) for f in [rp] + list(files) if f.is_file()])
    log('把 tfa_selfcheck.zip 拉回本地发我即可。')
    return zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', action='store_true',
                    help='在只读盘点之后额外跑 val/predict 导出（约 20~40 分钟）')
    ap.add_argument('--base', default=str(BASE), help='实验包搜索根')
    a = ap.parse_args()
    base = pathlib.Path(a.base)

    root, hits = find_root(base)
    if root is None:
        log('未找到 src/train_mvp.py；请用 --base 指定实验包所在根，例如')
        log('  python tfa_selfcheck.py --base /root/autodl-tmp')
        sys.exit(2)
    log('候选 train_mvp.py:', len(hits))
    rep = stage1(root)
    print_report(rep)
    rp = root / '输出' / 'tfa_scan_report.json'
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    log('盘点报告已写:', rp)

    if not a.export:
        log('（只读模式结束。加 --export 才会跑 val/predict 导出。）')
        return

    if rep['all_pt']['count'] == 0:
        log('=== 权重缺失：Stage 2 不执行，避免空跑 20-40 分钟 ===')
        log('走审读意见 §5.3：数据不删、解释降级为「未解释现象」、摘要删去；不需要重训。')
        pack(root, rp, [])
        return

    log('=== Stage 2 开始（不训练，只做 val / predict）===')
    result, files = run_export(root, rep)
    if result.get('fatal'):
        log('Stage 2 中止:', result['fatal'])
        pack(root, rp, files)
        return
    pack(root, rp, files)
    log('=== Stage 2 完成 ===')


if __name__ == '__main__':
    main()
