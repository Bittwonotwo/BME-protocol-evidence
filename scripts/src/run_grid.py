# -*- coding: utf-8 -*-
"""网格实验驱动: 固定"检测头宽度比例 s", 在不同标注量 N 上各跑一遍, 得到 mAP = f(s, N) 曲面。

为什么需要它:
    上一版只测了 N=10 和 N=1000 两个端点, 中间的曲面是空的。而"头容量该随标注量变"
    这个假设, 只有中间段才能验证。本脚本用固定比例扫描代替 w(N), 把"机制主张"先降级成
    "客观规律测量": 先看 s*(N) 到底怎么变, 再决定要不要拟合 w(N)。

用法:
    python src/run_grid.py                    # 默认 kill: N=10/30 x 4 档宽度 x 5 个抽样 = 40 个 run
    python src/run_grid.py --stage full       # 全网格: 再加 N=100/300/全量
    python src/run_grid.py --scales 0.25 1.0 --shots 10 --draws 3    # 自定义(便宜的一刀)
    python src/run_grid.py --dry-run          # 只打印计划, 不训练

说明:
    * 每个 run 的梯度更新次数固定为 800 (shot 档), 换 N 不改更新次数, 保证跨 N 可比。
    * 结果写 输出/runs_grid/, 汇总用 src/summarize_grid.py。
    * 断点续跑: 已有 final_metrics.json 的 run 自动跳过。
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _bootstrap  # noqa: E402

ROOT = HERE.parent
SRC = HERE
CFG_PATH = ROOT / "配置.txt"

# 每个 N 档单 run 的粗略耗时(秒), 仅用于打印预估, 4090D 实测量级
SEC_PER_RUN = {10: 70, 30: 130, 100: 220, 300: 380, 1000: 480}


def load_cfg(path):
    cfg = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def stage_plan(stage):
    """返回 (shots, scales, draws_map)"""
    scales = [0.25, 0.5, 0.75, 1.0]
    if stage == "kill":
        return [10, 30], scales, {10: 5, 30: 5}
    return [10, 30, 100, 300, 1000], scales, {10: 5, 30: 5, 100: 2, 300: 1, 1000: 1}


def epochs_for(n_images, batch, updates):
    """固定梯度更新次数 -> 换算成 epochs (N 小于 batch 时一 epoch 只有一个 batch)"""
    if n_images >= 900:          # 全量档沿用协议: 100 epoch
        return 100
    b = max(1, min(batch, n_images))
    batches = max(1, math.ceil(n_images / b))
    return max(1, round(updates / batches))


def scale_tag(s):
    return f"s{int(round(s * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["kill", "full"], default="kill")
    ap.add_argument("--shots", type=int, nargs="+", default=None,
                    help="覆盖档位: 例如 --shots 10 30")
    ap.add_argument("--scales", type=float, nargs="+", default=None,
                    help="覆盖宽度比例: 例如 --scales 0.25 0.5 0.75 1.0")
    ap.add_argument("--draws", type=int, default=None, help="覆盖每个 N 的抽样个数")
    ap.add_argument("--updates", type=int, default=800, help="shot 档固定梯度更新次数")
    ap.add_argument("--lr0", type=float, default=0.0,
                    help="覆盖初始学习率(默认 0.01)。做'窄头涨点是不是只因为更好调'的对照实验时用")
    ap.add_argument("--batch", type=int, default=0, help="0 = 读 配置.txt")
    ap.add_argument("--freeze", type=int, default=0, help="透传 train_mvp.py 的 --freeze, 0=不冻结")
    ap.add_argument("--device", default="", help="留空 = 读 配置.txt")
    ap.add_argument("--data-root", default="", help="留空 = 输出/mvp_data")
    ap.add_argument("--project", default="", help="留空 = 输出/runs_grid")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据划分(已准备好时用)")
    ap.add_argument("--dataset", default="", help="覆盖 配置.txt 的数据集目录 (换数据集用)")
    ap.add_argument("--names", default="", help="覆盖 配置.txt 的类别名, 逗号分隔 (换数据集用)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = ap.parse_args()

    cfg = load_cfg(CFG_PATH)
    ds = Path(args.dataset) if args.dataset else Path(cfg.get("数据集目录", "数据集"))
    if not ds.is_absolute():
        ds = ROOT / ds
    names = (args.names or cfg.get("类别名", "")).strip()
    batch = args.batch or int(cfg.get("batch", "8"))
    device = args.device or cfg.get("设备", "0").strip()
    out_data = Path(args.data_root) if args.data_root else ROOT / "输出" / "mvp_data"
    out_runs = Path(args.project) if args.project else ROOT / "输出" / "runs_grid"

    shots, scales, draws_map = stage_plan(args.stage)
    if args.shots:
        shots = args.shots
        draws_map = {n: (args.draws or draws_map.get(n, 3)) for n in shots}
    if args.scales:
        scales = args.scales
    if args.draws:
        draws_map = {n: args.draws for n in shots}

    if not (ds / "images").exists():
        raise SystemExit(f"[失败] {ds} 下没有 images 文件夹, 请检查 配置.txt 的 数据集目录")
    if not names:
        raise SystemExit("[失败] 配置.txt 里 类别名 为空")

    runs = []
    for n in shots:
        for seed in range(1, draws_map[n] + 1):
            split = "full" if n >= 900 else f"shot{n}_s{seed}"
            for s in scales:
                runs.append((n, s, split))

    total_sec = sum(SEC_PER_RUN.get(n, 300) for n, _, _ in runs)
    print("=" * 68)
    print(f"网格实验 stage={args.stage}")
    print(f"标注档位 N : {shots}   (每个 N 的抽样数 {draws_map})")
    print(f"宽度比例 s : {scales}")
    print(f"初始学习率 : {args.lr0 if args.lr0 > 0 else 0.01}"
          + ("  (覆盖)" if args.lr0 > 0 else ""))
    print(f"run 总数   : {len(runs)}   预计耗时 ≈ {total_sec / 60:.0f} 分钟 (4090D 量级)")
    print(f"结果目录   : {out_runs}")
    print("=" * 68)
    if args.dry_run:
        for n, s, split in runs:
            print(f"  {scale_tag(s)}  N={n:<5} {split:<14} epochs={epochs_for(n, batch, args.updates)}")
        return

    _bootstrap.ensure_cuda(require=True)

    # ---------- 1) 准备数据: 每个 N 档一次性生成其全部抽样 ----------
    if not args.skip_data:
        done_shots = set()
        for n in shots:
            if n in done_shots:
                continue
            done_shots.add(n)
            seeds = [str(i) for i in range(1, draws_map[n] + 1)]
            cmd = [sys.executable, str(SRC / "prepare_data.py"),
                   "--src", str(ds), "--out", str(out_data), "--names", names,
                   "--shots", str(n), "--sample-seeds", *seeds]
            sf = ds / "official_split.txt"
            if sf.exists():
                cmd += ["--split-file", str(sf)]
            print(f"\n[数据] N={n} 抽样种子 {seeds}")
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                raise SystemExit(f"[失败] 数据准备出错 (N={n}), 修复后重跑本脚本即可")

    # ---------- 2) 逐 run 训练 ----------
    out_runs.mkdir(parents=True, exist_ok=True)
    ok, skipped, failed = [], [], []
    batch_now = batch
    t0 = time.time()
    for i, (n, s, split) in enumerate(runs, 1):
        name = f"grid_{scale_tag(s)}_{split}"
        marker = out_runs / name / "final_metrics.json"
        ep = epochs_for(n, batch_now, args.updates)
        if marker.exists():
            print(f"[{i}/{len(runs)}] {name} 已完成, 跳过")
            skipped.append(name)
            continue
        print(f"\n[{i}/{len(runs)}] ===== {name}  N={n} s={s} epochs={ep} =====")
        attempt = 0
        while True:
            cmd = [sys.executable, str(SRC / "train_mvp.py"),
                   "--group", "g0", "--split", split, "--run-name", name,
                   "--data-root", str(out_data), "--project", str(out_runs),
                   "--device", device, "--batch", str(batch_now),
                   "--epochs", str(ep), "--head-scale", str(s)]
            if args.lr0 > 0:
                cmd += ["--lr0", str(args.lr0)]
            if args.freeze > 0:
                cmd += ["--freeze", str(args.freeze)]
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode == 0:
                ok.append(name)
                break
            if batch_now <= 1 or attempt >= 2:
                failed.append(name)
                break
            attempt += 1
            batch_now = max(1, batch_now // 2)
            print(f"\n[重试] {name} 失败(多为显存不足), 改用 batch={batch_now} 重跑 (第 {attempt} 次)")
            import shutil
            shutil.rmtree(out_runs / name, ignore_errors=True)
            time.sleep(3)
        el = (time.time() - t0) / 60
        left = sum(SEC_PER_RUN.get(nn, 300) for nn, _, _ in runs[i:]) / 60
        print(f"[进度] {i}/{len(runs)}  已用 {el:.0f} 分钟, 预计剩余 ≈ {left:.0f} 分钟")

    print("\n========== 网格执行汇总 ==========")
    print(f"新完成 {len(ok)} 个, 跳过 {len(skipped)} 个, 失败 {len(failed)} 个")
    if failed:
        print(f"失败列表: {failed}")
        print("修复后重跑本脚本会从断点继续。")
    else:
        print("下一步: python src/summarize_grid.py   (生成 输出/网格结果.md)")


if __name__ == "__main__":
    main()
