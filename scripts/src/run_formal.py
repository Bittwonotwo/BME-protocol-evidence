# -*- coding: utf-8 -*-
"""步骤3: 正式实验 —— 读取 配置.txt, 生成数据划分, 跑 8/16 个 run。
中断后重新运行本脚本会从断点继续(已完成的 run 自动跳过)。

运行: python src/run_formal.py    (或 python run.py formal)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CFG_PATH = ROOT / "配置.txt"


def load_cfg(path):
    cfg = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def run(args):
    print("\n>>>", " ".join(str(a) for a in args[-6:]))
    r = subprocess.run([sys.executable, *[str(a) for a in args]], cwd=str(ROOT))
    if r.returncode != 0:
        print("\n[失败] 上一步出错, 已中止。修复后重新运行 `python run.py formal` 即可从断点继续。")
        sys.exit(1)


def main():
    # 正式实验必须有 GPU: 用 CPU 跑要几天, 所以这里直接把话说死, 不静默降级
    _bootstrap.ensure_cuda(require=True)

    cfg = load_cfg(CFG_PATH)
    ds = Path(cfg.get("数据集目录", "数据集"))
    if not ds.is_absolute():
        ds = ROOT / ds
    names = cfg.get("类别名", "").strip()
    batch = int(cfg.get("batch", "8"))
    device = cfg.get("设备", "0").strip()
    tier = cfg.get("实验档位", "快速").strip()
    epochs = int(cfg.get("训练轮数", "0") or 0)
    quick = tier.startswith("快速")
    splits = ["full", "shot10_s1"] if quick else \
             ["full", "shot10_s1", "shot10_s2", "shot10_s3"]

    print("=" * 60)
    print(f"数据集: {ds}")
    print(f"类别名: {names}")
    print(f"batch : {batch}   设备: {device}")
    print(f"档位  : {'快速判定 (' + str(len(splits) * 4) + ' 个训练)' if quick else '标准 (' + str(len(splits) * 4) + ' 个训练)'}"
          f"   训练轮数: {'自动' if epochs == 0 else epochs}")
    print("=" * 60)

    if not (ds / "images").exists():
        raise SystemExit(
            f"[失败] 在 {ds} 下没找到 images 文件夹。\n"
            "请把数据集整理成: 数据集/images/*.jpg 与 数据集/labels/*.txt (YOLO 格式),\n"
            "然后在 配置.txt 里填对 数据集目录。"
        )
    if not names or "例如" in names:
        raise SystemExit("[失败] 请先在 配置.txt 里填写 类别名 (英文逗号分隔, 顺序与标注类别 id 一致)")

    out_data = ROOT / "输出" / "mvp_data"
    out_runs = ROOT / "输出" / "runs_mvp"

    prep = [SRC / "prepare_data.py", "--src", ds, "--out", out_data, "--names", names]
    split_file = ds / "official_split.txt"
    if split_file.exists():
        print("检测到 official_split.txt -> 使用该划分（DeepPCB 官方 1000/500，与文献可比）")
        prep += ["--split-file", str(split_file)]
    run(prep)
    cmd = [SRC / "run_all.py", "--data-root", out_data, "--project", out_runs,
           "--batch", batch, "--device", device, "--splits", *splits]
    if epochs > 0:
        cmd += ["--epochs", epochs]
    run(cmd)

    print("\n" + "=" * 60)
    print(f"{len(splits) * 4} 个 run 全部完成。下一步: python run.py summary")
    if quick:
        print("当前是快速判定档(10-shot 只用了 1 个抽样种子)。若结论为通过, 把 配置.txt 里的")
        print("「实验档位」改成 标准 再运行 python run.py formal, 会补跑另外 2 个种子"
              "(已有结果自动跳过, 不重复训练)。")
    print("=" * 60)


if __name__ == "__main__":
    main()
