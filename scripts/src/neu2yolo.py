# -*- coding: utf-8 -*-
"""把 NEU-DET（钢材表面缺陷）转成本实验包要求的 YOLO 格式。

NEU-DET 两种常见发行结构都已兼容（按文件名匹配，不用管目录长什么样）：
    官方原版:  IMAGES/crazing_1.jpg ...  ANNOTATIONS/crazing_1.xml ...   (平铺)
    Kaggle 版: train/images/<类名>/*.jpg + train/annotations/*.xml
               validation/images/...  + validation/annotations/...

输出（与 DeepPCB 转换后的格式一致，可直接被 prepare_data.py 消费）：
    <out>/images/*.jpg
    <out>/labels/*.txt     (YOLO 归一化: class cx cy w h)

注意 3 个坑（已处理）：
    1. 无官方 train/test 划分 —— 不生成 official_split.txt，由 prepare_data.py
       按固定 seed 做 70/30（跨数据集口径一致，论文里必须写明这一点）；
    2. XML 的 <name> 写法不统一（"rolled-in scale" / "rolled-in_scale" / 大小写），
       统一归一化：小写 + 空格/连字符 -> 下划线；
    3. 图像是 200x200 灰度图 —— 归一化直接用 XML 里的 <size>，缺失时回退 200。

用法:
    python src/neu2yolo.py --src /root/autodl-tmp/NEU-DET --out 数据集_NEU
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def norm(s: str) -> str:
    """类名归一化：小写，空格与连字符变下划线。"""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


NORM2ID = {norm(c): i for i, c in enumerate(CLASS_NAMES)}


def parse_xml(p: Path):
    """返回 (width, height, [(cls_id, cx, cy, w, h), ...])；解析失败返回 None。"""
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return None
    size = root.find("size")
    if size is not None and size.find("width") is not None:
        w_img = float(size.find("width").text)
        h_img = float(size.find("height").text)
    else:
        w_img = h_img = 200.0
    boxes = []
    for obj in root.iter("object"):
        name = obj.find("name")
        bb = obj.find("bndbox")
        if name is None or bb is None:
            continue
        cid = NORM2ID.get(norm(name.text or ""))
        if cid is None:
            continue
        try:
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
        except (AttributeError, ValueError):
            continue
        cx = min(max((x1 + x2) / 2 / w_img, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2 / h_img, 0.0), 1.0)
        w = min(max((x2 - x1) / w_img, 0.0), 1.0)
        h = min(max((y2 - y1) / h_img, 0.0), 1.0)
        if w > 0 and h > 0:
            boxes.append((cid, cx, cy, w, h))
    return w_img, h_img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="NEU-DET 目录（官方原版或 Kaggle 版均可）")
    ap.add_argument("--out", default="数据集_NEU",
                    help="输出目录（生成 images/ labels/）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"[失败] 找不到目录 {src}")

    xmls = {p.stem: p for p in src.rglob("*.xml")}
    imgs = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.setdefault(p.stem, p)
    if not xmls:
        raise SystemExit(f"[失败] {src} 下没有 .xml 标注文件，请确认 --src 指向数据根目录")

    out = Path(args.out)
    d_img, d_lbl = out / "images", out / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)

    n_ok = n_noimg = n_empty = 0
    per_class = {c: 0 for c in CLASS_NAMES}
    for stem, xp in sorted(xmls.items()):
        ip = imgs.get(stem)
        if ip is None:
            n_noimg += 1
            continue
        parsed = parse_xml(xp)
        if parsed is None:
            n_empty += 1
            continue
        _, _, boxes = parsed
        if not boxes:
            n_empty += 1
            continue
        shutil.copy2(ip, d_img / ip.name)
        with open(d_lbl / f"{stem}.txt", "w", encoding="utf-8") as f:
            for cid, cx, cy, w, h in boxes:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                per_class[CLASS_NAMES[cid]] += 1
        n_ok += 1

    print(f"[完成] 转换 {n_ok} 张 -> {out}")
    if n_noimg:
        print(f"[警告] {n_noimg} 个 XML 找不到同名图片，已跳过")
    if n_empty:
        print(f"[警告] {n_empty} 个 XML 解析为空/失败，已跳过")
    print("[类别框数] " + "  ".join(f"{c}={n}" for c, n in per_class.items()))
    print(f"[类名顺序] {','.join(CLASS_NAMES)}")
    print("下一步（划分+训练，一条命令）:")
    print(f"  python src/run_grid.py --dataset {args.out} "
          f"--names {','.join(CLASS_NAMES)} "
          f"--data-root 输出/mvp_data_neu --project 输出/runs_neu "
          f"--shots 10 30 100 --scales 1.0 --batch 8 --lr0 0.01")


if __name__ == "__main__":
    main()
