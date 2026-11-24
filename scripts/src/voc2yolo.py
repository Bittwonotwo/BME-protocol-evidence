# -*- coding: utf-8 -*-
"""工具: 把 Pascal VOC 格式 (XML 标注) 转成 YOLO 格式 (txt)。
DeepPCB / NEU-DET 的原始标注常见为 XML, 若你拿到的是 XML 版本就用这个脚本。

用法示例:
    python src/voc2yolo.py --src 原始数据 --out 数据集 \
        --classes "missing_hole,mouse_bite,open_circuit,short,spur,spurious_copper"

要求原始数据里有 Annotations/*.xml, 图片放在任意位置(脚本按文件名匹配 jpg/png)。
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="", help="逗号分隔, 顺序即类别 id; 留空则按 XML 中的类名字母序")
    args = ap.parse_args()

    src = Path(args.src)
    xmls = list(src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"在 {src} 下没有找到 xml 标注")
    imgs = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS}

    if args.classes.strip():
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        found = set()
        for x in xmls:
            for obj in ET.parse(x).getroot().iter("object"):
                found.add(obj.findtext("name", "").strip())
        classes = sorted(c for c in found if c)
    cid = {c: i for i, c in enumerate(classes)}

    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_ok, n_miss, n_skip = 0, 0, 0
    for x in xmls:
        root = ET.parse(x).getroot()
        stem = x.stem
        im = imgs.get(stem)
        if im is None:
            n_miss += 1
            continue
        w = float(root.findtext("size/width", "0") or 0)
        h = float(root.findtext("size/height", "0") or 0)
        if w <= 0 or h <= 0:
            n_skip += 1
            continue
        lines = []
        for obj in root.iter("object"):
            name = obj.findtext("name", "").strip()
            if name not in cid:
                continue
            b = obj.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cid[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        shutil.copy2(im, out_img / im.name)
        (out_lbl / f"{im.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_ok += 1

    print(f"转换完成: 成功 {n_ok} 张, 找不到图片 {n_miss}, 缺尺寸信息跳过 {n_skip}")
    print(f"类别顺序: {classes}")
    print(f"输出: {Path(args.out)}/images 与 /labels")
    print("下一步: 在 配置.txt 里把 数据集目录 指向该输出目录, 类别名保持同样顺序")


if __name__ == "__main__":
    main()
