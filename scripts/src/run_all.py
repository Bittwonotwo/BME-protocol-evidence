# -*- coding: utf-8 -*-
"""
MVP 全量驱动: 4 组 x 4 split = 16 个 run, 顺序执行, 断点续跑(已有 final_metrics.json 的自动跳过)。

用法:
    python run_all.py                # 正式 16 个 run (shot 档 800 epoch, 全量档 100 epoch)
    python run_all.py --epochs 3     # 冒烟测试 (配合 dummy 数据, 几分钟跑完全链路)
    python run_all.py --groups g1 g2 --splits shot10_s1   # 只跑部分
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

GROUP_LABELS = {
    "g0": "baseline",
    "g1": "+频域注意力",
    "g2": "+样本量自适应头",
    "g3": "+两者组合",
}


def _expected_epochs(args, split):
    return args.epochs or (800 if split.startswith("shot") else 100)


def _marker_ok(marker, expected_epochs):
    """续跑标记不仅要存在，训练轮数还得和本次设置一致——
    防止把早期测试留下的 1-epoch 结果当成正式结果跳过。"""
    try:
        import json
        d = json.loads(marker.read_text(encoding="utf-8"))
        return d.get("config", {}).get("epochs") == expected_epochs
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["g0", "g1", "g2", "g3"])
    ap.add_argument("--splits", nargs="+", default=["full", "shot10_s1", "shot10_s2", "shot10_s3"])
    ap.add_argument("--data-root", default="mvp_data")
    ap.add_argument("--project", default="runs_mvp")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8, help="3060 6G 笔记本用 8(默认), 12G 及以上可 16")
    ap.add_argument("--epochs", type=int, default=0, help="0 = 自动 (shot 800 / full 100)")
    ap.add_argument("--train-val", action="store_true",
                    help="强制所有档位训练中验证(慢, 默认 shot 档关闭)")
    args = ap.parse_args()

    data_root = str(Path(args.data_root).resolve())
    project = str(Path(args.project).resolve())

    runs = [(g, s) for g in args.groups for s in args.splits]
    print(f"共 {len(runs)} 个 run, 项目目录 {project}, 起始 batch={args.batch}")
    done, failed, skipped = [], [], []
    batch = args.batch
    for i, (g, s) in enumerate(runs, 1):
        marker = Path(project) / f"{g}_{s}" / "final_metrics.json"
        if marker.exists():
            exp = _expected_epochs(args, s)
            if _marker_ok(marker, exp):
                print(f"[{i}/{len(runs)}] {g}_{s} 已完成 (epochs={exp}), 跳过")
                skipped.append(f"{g}_{s}")
                continue
            print(f"[{i}/{len(runs)}] {g}_{s} 有旧结果但训练轮数不符 (需要 {exp} 轮), 重跑")
            shutil.rmtree(marker.parent, ignore_errors=True)
        print(f"\n[{i}/{len(runs)}] ===== {g} ({GROUP_LABELS.get(g, '')}) x {s} =====")
        # 显存保护: 某一组因 OOM 失败时, 换新进程用一半的 batch 重跑, 后续组沿用新 batch。
        # (OOM 会让 CUDA 上下文损坏, 在同进程里重试往往报 "invalid argument", 必须换进程)
        attempt = 0
        while True:
            cmd = [
                sys.executable, str(HERE / "train_mvp.py"),
                "--group", g, "--split", s,
                "--data-root", data_root,
                "--project", project,
                "--device", args.device,
                "--batch", str(batch),
            ]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.train_val:
                cmd += ["--train-val"]
            r = subprocess.run(cmd)
            if r.returncode == 0:
                done.append(f"{g}_{s}")
                break
            if batch <= 1 or attempt >= 2:
                failed.append(f"{g}_{s}")
                break
            attempt += 1
            batch = max(1, batch // 2)
            print(f"\n[重试] {g}_{s} 失败(多为显存不足), 改用 batch={batch} 重跑 "
                  f"(第 {attempt} 次重试), 后续各组也用这个 batch。")
            shutil.rmtree(Path(project) / f"{g}_{s}", ignore_errors=True)
            time.sleep(3)

    print("\n========== 执行汇总 ==========")
    print(f"新完成: {len(done)} {done}")
    print(f"跳过(此前已完成): {len(skipped)}")
    print(f"失败: {len(failed)} {failed}")
    if failed:
        print("请检查上方报错; 修复后重跑 run_all.py 会自动跳过已完成的 run。")
    else:
        print("全部完成, 下一步: python summarize.py")


if __name__ == "__main__":
    main()
