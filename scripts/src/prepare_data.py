# -*- coding: utf-8 -*-
"""
数据准备: 70/30 划分 + 10-shot 分层子采样 (3 个采样种子) + 生成各 split 的 YOLO 数据集 yaml

输入目录 (--src) 需为 YOLO 格式:
    <src>/images/**/*.{jpg,png,...}
    <src>/labels/**/<同名>.txt     (每行: cls xc yc w h, 归一化; 无标签文件视为纯背景图)

输出目录 (--out):
    <out>/test/            密封测试集 (所有 run 共用)
    <out>/full/            全量训练池 (70%)
    <out>/shot10_s{1,2,3}/ 10-shot 子集 (分层采样, 每个 seed 一份)
    <out>/<split>.yaml     每个 split 一个 yaml, val 一律指向 test
    <out>/meta.json        采样审计信息
"""
import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(src: Path):
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    labels = {p.stem: p for p in src.rglob("*.txt")}
    pairs = []
    for im in sorted(imgs, key=lambda p: p.stem):
        pairs.append((im, labels.get(im.stem)))
    return pairs


def label_classes(lb_path: Path):
    out = []
    for line in lb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            out.append(int(float(line.split()[0])))
    return out


def write_split(out_root: Path, name: str, pairs, names):
    d_img = out_root / name / "images"
    d_lbl = out_root / name / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)
    n_obj = 0
    for im, lb in pairs:
        shutil.copy2(im, d_img / im.name)
        if lb is not None and lb.exists():
            shutil.copy2(lb, d_lbl / lb.name)
            n_obj += len(label_classes(lb))
    yaml_text = (
        f"path: {out_root.resolve().as_posix()}\n"
        f"train: {name}/images\n"
        f"val: test/images\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out_root / f"{name}.yaml").write_text(yaml_text, encoding="utf-8")
    return {"n_images": len(pairs), "n_objects": n_obj}


def sample_shots(train_pairs, shots, seed, n_classes):
    """分层采样: 类别均衡 + 每类尽量至少 1 张 (类数 <= shots 时), 去重, 总数不超过 shots"""
    rng = random.Random(seed)
    cls2imgs = {c: [] for c in range(n_classes)}
    inst_count = {c: 0 for c in range(n_classes)}
    for im, lb in train_pairs:
        if lb is None:
            continue
        for c in label_classes(lb):
            if c in cls2imgs:
                cls2imgs[c].append(im)
                inst_count[c] += 1

    empty_cls = [c for c in cls2imgs if not cls2imgs[c]]
    if empty_cls:
        print(f"[警告] 以下类别在训练池中没有出现, 无法参与采样: {empty_cls}")

    # 配额: 按实例数比例, 类数 <= shots 时每类保底 1 张
    active = [c for c in range(n_classes) if cls2imgs[c]]
    quota = {c: 0 for c in range(n_classes)}
    if not active:
        raise RuntimeError("训练池中没有任何带标注的图像")
    total_inst = sum(inst_count[c] for c in active)
    for c in active:
        q = shots * inst_count[c] / total_inst
        if len(active) <= shots:
            q = max(1.0, q)
        quota[c] = int(q)
    # 调整配额总和到 shots
    order = sorted(active, key=lambda c: (-(quota[c] - int(quota[c])), c))
    diff = shots - sum(quota.values())
    i = 0
    while diff != 0 and order:
        c = order[i % len(order)]
        if diff > 0:
            quota[c] += 1
            diff -= 1
        elif quota[c] > 1:
            quota[c] -= 1
            diff += 1
        i += 1

    picked, seen = [], set()
    for c in active:
        pool = sorted(cls2imgs[c], key=lambda p: p.stem)
        rng.shuffle(pool)
        for im in pool[: quota[c]]:
            if im.stem not in seen:
                seen.add(im.stem)
                picked.append(im)
    picked = picked[:shots]
    # 去重后不足 shots 时, 从训练池剩余带标注图像中随机补齐
    if len(picked) < shots:
        rest = sorted(
            [im for im, lb in train_pairs if lb is not None and im.stem not in seen],
            key=lambda p: p.stem,
        )
        rng.shuffle(rest)
        for im in rest[: shots - len(picked)]:
            seen.add(im.stem)
            picked.append(im)
    stem2pair = {im.stem: (im, lb) for im, lb in train_pairs}
    return [stem2pair[im.stem] for im in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="YOLO 格式源数据目录 (含 images/ 与 labels/)")
    ap.add_argument("--out", default="mvp_data", help="输出目录")
    ap.add_argument("--names", required=True, help='逗号分隔的类别名, 如 "hole,mouse_bite,open"')
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=0, help="70/30 划分种子, 全程固定")
    ap.add_argument("--split-file", default="",
                    help="可选: 每行 '图片文件名<TAB>train|test'。提供则用该划分替代随机 70/30 "
                         "(DeepPCB 转换脚本会生成 official_split.txt, 用它可保证与文献可比)")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--sample-seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    n_classes = len(names)
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"在 {src} 下没有找到图片, 请检查目录结构")

    # 1) 划分: 有官方划分文件就用它（保证与文献可比），否则 70/30 随机（固定种子，全程不变）
    assign = {}
    split_file = Path(args.split_file) if args.split_file else None
    if split_file is not None:
        if not split_file.is_absolute():
            # 相对路径: 先按当前目录找, 再按源数据目录找（两种调用习惯都支持）
            for cand in (Path.cwd() / split_file, src / split_file):
                if cand.exists():
                    split_file = cand
                    break
            else:
                split_file = src / split_file
        if split_file.exists():
            for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.replace("\t", " ").split()
                if len(parts) >= 2:
                    assign[parts[0]] = parts[1].lower()
        else:
            print(f"[警告] 找不到划分文件 {split_file}, 改用 70/30 随机划分")

    if assign:
        test_pairs = [p for p in pairs if assign.get(p[0].name) == "test"]
        train_pairs = [p for p in pairs if assign.get(p[0].name) != "test"]
        unassigned = sum(1 for p in pairs if p[0].name not in assign)
        if unassigned:
            print(f"[警告] {unassigned} 张图未在划分文件中列出, 已归入训练池")
        print(f"使用指定的划分文件: 训练池 {len(train_pairs)} / 测试集 {len(test_pairs)}")
    else:
        rng = random.Random(args.split_seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_test = round(len(pairs) * args.test_ratio)
        test_pairs = [pairs[i] for i in idx[:n_test]]
        train_pairs = [pairs[i] for i in idx[n_test:]]
        print(f"总计 {len(pairs)} 图 -> 训练池 {len(train_pairs)} / 测试集 {len(test_pairs)}")
    if not test_pairs or not train_pairs:
        raise SystemExit("[失败] 划分后训练池或测试集为空, 请检查 --split-file 内容")

    out.mkdir(parents=True, exist_ok=True)
    meta = {"names": names, "split_seed": args.split_seed, "splits": {}}

    meta["splits"]["test"] = write_split(out, "test", test_pairs, names)
    meta["splits"]["full"] = write_split(out, "full", train_pairs, names)

    for seed in args.sample_seeds:
        shot_pairs = sample_shots(train_pairs, args.shots, seed, n_classes)
        name = f"shot{args.shots}_s{seed}"
        meta["splits"][name] = write_split(out, name, shot_pairs, names)
        per_class = {}
        for _, lb in shot_pairs:
            if lb is None:
                continue
            for c in label_classes(lb):
                per_class[c] = per_class.get(c, 0) + 1
        meta["splits"][name]["objects_per_class"] = per_class
        print(f"{name}: {len(shot_pairs)} 图, 每类实例数 {per_class}")

    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成。数据集 yaml 位于 {out}/, 接下来可运行 run_all.py")


if __name__ == "__main__":
    main()
