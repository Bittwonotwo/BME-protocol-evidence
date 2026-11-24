# -*- coding: utf-8 -*-
"""PDC 前提验证：跑 6 个配置，记录参数位移曲线，与已知 mAP 对照。

要回答的唯一问题：
    delta = ||theta_t - theta_0|| / ||theta_0|| 能否预测最终 mAP？
      - 能   -> 可以用 delta 做在线 lr 控制器（论文的方法章成立）
      - 不能 -> 放弃方法路线，收成评测协议论文（BME）

为什么这 6 格就够：
    它们是"同一批数据、同一 800 次更新预算、只改 batch 与 lr"的 3×2 网格，
    mAP 已知跨度从 79.02 到 92.85，足够看 delta 是否随 mAP 单调。
    关键一对是 (N=30, b=30) 的 lr=0.01 (82.33) 与 lr=0.002 (85.19)：
    两者数据完全相同、只有 lr 不同，若 delta 能把它们分开且方向一致，
    delta 就是有效可观测量。

用法：
    python src/run_delta.py              # 6 个 run，约 15-25 分钟
    python src/run_delta.py --draw 2     # 换抽样种子
    python src/summarize_delta.py        # 出对照表 + 判定
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# (N, batch, lr, 参照 mAP50) —— 参照值来自已完成的 batch 对照实验
GRID = [
    (30, 30, 0.01, 82.33),
    (30, 30, 0.002, 85.19),
    (30, 8, 0.01, 82.28),
    (30, 8, 0.002, 79.02),
    (100, 32, 0.01, 92.85),
    (100, 8, 0.01, 87.68),
]


def epochs_for(n, b, updates):
    """固定更新次数换算 epochs：epochs = updates / (N/batch)"""
    per_epoch = math.ceil(n / b)
    return max(1, round(updates / per_epoch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="输出/runs_delta")
    ap.add_argument("--data-root", default="输出/mvp_data")
    ap.add_argument("--every", type=int, default=50, help="每多少步记录一次位移")
    ap.add_argument("--draw", type=int, default=1, help="抽样种子编号 (1..10)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--updates", type=int, default=800)
    args = ap.parse_args()

    proj = Path(args.project)
    proj.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # 数据档必须已存在（这些档在之前的 batch 对照实验里已经生成过）
    need = sorted({f"shot{n}_s{args.draw}.yaml" for n, _, _, _ in GRID})
    missing = [f for f in need if not (data_root / f).exists()]
    if missing:
        raise SystemExit(
            f"[数据缺失] {data_root} 下找不到: {', '.join(missing)}\n"
            f"这些档在 batch 对照实验里应当已生成；若确实没有，先跑一次对应的 run_grid。"
        )

    done = skipped = failed = 0
    failed_list = []
    for n, b, lr, ref in GRID:
        ep = epochs_for(n, b, args.updates)
        name = f"delta_N{n}_b{b}_lr{lr:g}"
        if (proj / name / "final_metrics.json").exists():
            print(f"[跳过] {name} 已完成")
            skipped += 1
            continue
        split = f"shot{n}_s{args.draw}"
        cmd = [
            sys.executable, str(SRC / "train_mvp.py"),
            "--group", "g0", "--split", split,
            "--data-root", str(data_root),
            "--project", str(proj),
            "--epochs", str(ep), "--batch", str(b),
            "--lr0", str(lr), "--head-scale", "1.0",
            "--record-delta", str(args.every),
            "--run-name", name, "--device", args.device,
        ]
        print(f"\n{'='*70}\n[N={n} batch={b} lr={lr}] epochs={ep} "
              f"(约 {ep * math.ceil(n / b)} 步)  参照 mAP50={ref}\n{'='*70}")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode == 0:
            done += 1
        else:
            failed += 1
            failed_list.append(name)
            print(f"[失败] {name} 返回码 {r.returncode}（重跑本脚本会自动续跑）")

    print(f"\n{'='*70}")
    print(f"完成 {done} / 跳过 {skipped} / 失败 {failed}")
    if failed_list:
        print("失败列表: " + ", ".join(failed_list))
    print("下一步: python src/summarize_delta.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
