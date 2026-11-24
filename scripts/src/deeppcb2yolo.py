# -*- coding: utf-8 -*-
"""把 DeepPCB 原始数据转成本实验包要求的 YOLO 格式。

DeepPCB 原始结构 (git clone 后):
    PCBData/
    ├── group00041/
    │   ├── 00041/            00041000_temp.jpg (无缺陷模板), 00041000_test.jpg (待检测图)
    │   └── 00041_not/        00041000.txt     (标注: x1 y1 x2 y2 type)
    ├── trainval.txt          官方训练划分 (1000 条)
    └── test.txt              官方测试划分 (500 条)

注意 3 个容易踩的坑 (已在本脚本处理):
    1. 标注是空格分隔的 "x1 y1 x2 y2 type", 不是 README 里写的逗号;
    2. trainval.txt / test.txt 里的图片名省略了 "_test" 后缀 (写 20085000.jpg, 实际是 20085000_test.jpg);
    3. type 从 1 开始 (1=open ... 6=pin-hole), YOLO 的类别 id 从 0 开始, 需要减 1。

用法:
    python src/deeppcb2yolo.py --src <DeepPCB 目录> --out 数据集
"""
import argparse
import shutil
from pathlib import Path

# type 1..6 -> YOLO 类别 0..5
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pcbdata(src: Path) -> Path:
    """定位 PCBData 目录: 允许 --src 指向仓库根目录或 PCBData 本身。"""
    if (src / "PCBData").is_dir():
        return src / "PCBData"
    if src.name == "PCBData" and src.is_dir():
        return src
    hits = list(src.rglob("PCBData"))
    hits = [h for h in hits if (h / "group00041").is_dir()]
    if hits:
        return hits[0]
    raise SystemExit(
        f"[失败] 在 {src} 下没找到 PCBData 目录。\n"
        "请确认 --src 指向 DeepPCB 仓库根目录（里面有 PCBData 文件夹）。"
    )


def read_split_list(path: Path):
    """读取 trainval.txt / test.txt -> {图片文件名(带 _test.jpg): 'train'|'test'}"""
    out = {}
    tag = "test" if "test" in path.stem.lower() else "train"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        img_rel = line.split()[0]              # 形如 group20085/20085/20085000.jpg
        stem = Path(img_rel).stem              # 20085000（缺 _test 后缀）
        # 官方列表写的是 20085000.jpg，真实文件是 20085000_test.jpg
        out[f"{stem}_test.jpg"] = tag
    return out


def img_size(p: Path):
    """用 PIL 读尺寸（cv2.imread 在中文路径下会失败，这里不用 cv2）。"""
    from PIL import Image
    with Image.open(p) as im:
        return im.size   # (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DeepPCB 仓库根目录（含 PCBData）")
    ap.add_argument("--out", default="数据集", help="输出目录（生成 images/ labels/ official_split.txt）")
    args = ap.parse_args()

    src = Path(args.src)
    pcb = find_pcbdata(src)
    out = Path(args.out)
    d_img = out / "images"
    d_lbl = out / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)
    print(f"[源] {pcb}")
    print(f"[目标] {out}")

    test_imgs = sorted(p for p in pcb.rglob("*_test.jpg"))
    if not test_imgs:
        raise SystemExit("[失败] 没找到任何 *_test.jpg，请检查 PCBData 是否完整")

    n_ok, n_skip, n_box = 0, 0, 0
    cls_count = {c: 0 for c in range(len(CLASS_NAMES))}
    for im in test_imgs:
        # 同级目录的 <group>_not/ 下放标注
        lbl_dir = im.parent.parent / f"{im.parent.name}_not"
        lbl = lbl_dir / f"{im.stem[:-len('_test')]}.txt"
        if not lbl.exists():
            n_skip += 1
            continue
        w, h = img_size(im)
        lines = []
        for raw in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = raw.strip().replace(",", " ")
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in parts[:4])
                t = int(float(parts[4]))
            except ValueError:
                continue
            if not (1 <= t <= 6):
                continue
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{t - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            cls_count[t - 1] += 1
            n_box += 1
        if not lines:
            n_skip += 1
            continue
        shutil.copy2(im, d_img / im.name)
        (d_lbl / f"{im.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1

    # 官方划分（1000 训练 / 500 测试）—— 保留下来，让实验与文献可比
    split = {}
    for name in ("trainval.txt", "test.txt"):
        f = pcb / name
        if f.exists():
            split.update(read_split_list(f))
    split = {k: v for k, v in split.items() if (d_img / k).exists()}
    if split:
        sf = out / "official_split.txt"
        sf.write_text("\n".join(f"{k}\t{v}" for k, v in sorted(split.items())) + "\n",
                      encoding="utf-8")
        n_tr = sum(1 for v in split.values() if v == "train")
        n_te = len(split) - n_tr
        print(f"[划分] official_split.txt: 训练 {n_tr} / 测试 {n_te}（官方划分）")
    else:
        print("[划分] 未找到 trainval.txt / test.txt，将按 70/30 随机划分")

    print(f"[完成] 转换 {n_ok} 张图 / {n_box} 个框；跳过 {n_skip} 张（无标注或标注为空）")
    print("[类别] " + "  ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)))
    print("[每类框数] " + "  ".join(f"{CLASS_NAMES[i]}:{c}" for i, c in cls_count.items()))
    print(f"\n下一步: 检查 配置.txt 的 类别名 写成:\n  {','.join(CLASS_NAMES)}")
    print("然后运行: python run.py formal")


if __name__ == "__main__":
    main()
