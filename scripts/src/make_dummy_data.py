# -*- coding: utf-8 -*-
"""生成 120 张合成缺陷图 (2 类, 随机小矩形), 用于冒烟测试全链路。真实数据就绪前的演练用。"""
import argparse
import random

import cv2
import numpy as np

CLASSES = ["defect_a", "defect_b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dummy_src")
    args = ap.parse_args()
    random.seed(0)
    np.random.seed(0)
    out_img = f"{args.out}/images"
    out_lbl = f"{args.out}/labels"
    import os
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    for i in range(120):
        img = np.random.randint(40, 90, (640, 640, 3), dtype=np.uint8)
        lines = []
        for _ in range(random.randint(1, 4)):
            c = random.randrange(2)
            w = random.randint(12, 60)
            h = random.randint(12, 60)
            x = random.randint(10, 630 - w)
            y = random.randint(10, 630 - h)
            color = (200, 180, 160) if c == 0 else (60, 80, 200)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            xc, yc = (x + w / 2) / 640, (y + h / 2) / 640
            lines.append(f"{c} {xc:.6f} {yc:.6f} {w / 640:.6f} {h / 640:.6f}")
        # 注意: cv2.imwrite 在 Windows 下不支持中文路径(静默失败), 必须走 imencode + tofile
        ok, buf = cv2.imencode(".jpg", img)
        buf.tofile(f"{out_img}/img_{i:04d}.jpg")
        with open(f"{out_lbl}/img_{i:04d}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"已生成 {args.out}/ (120 图, 2 类)")


if __name__ == "__main__":
    main()
