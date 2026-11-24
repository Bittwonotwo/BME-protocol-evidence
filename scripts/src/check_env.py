# -*- coding: utf-8 -*-
"""步骤1: 环境检查 —— python / torch / CUDA / 显卡 / 显存, 并给出 batch 建议

直接用 python 运行即可:
    python src/check_env.py
若当前解释器的 torch 用不了 GPU, 会自动切换到本机的 CUDA 环境再执行(见 _bootstrap.py)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402  必须在 import torch 之前

_bootstrap.ensure_cuda(require=False)


def main():
    print(f"解释器: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"torch: {torch.__version__}")
    except ImportError:
        print("[失败] 未安装 torch, 请先运行: python -m pip install -r requirements.txt")
        return
    try:
        import ultralytics
        print(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("[失败] 未安装 ultralytics")
        return

    print("-" * 50)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        print(f"[OK] CUDA 可用, 显卡: {name}, 显存 {total:.1f} GB (当前空闲 {free:.1f} GB)")
        print(f"CUDA 版本: {torch.version.cuda}")
        if total >= 20:
            print("建议 batch = 32 (配置.txt 已按 4090D 24G 设为 32)")
        elif total >= 10:
            print("建议 batch = 16 (把 配置.txt 里的 batch 改成 16)")
        elif total >= 5.5:
            print("建议 batch = 8")
        else:
            print("建议 batch = 4  (请把 配置.txt 里的 batch 改成 4)")
        print(f"预计耗时: 快速档 8 个训练约 {40 if total >= 5.5 else 70} 分钟"
              f"(6GB 实测: 10-shot 档 1 个约 8.5 分钟, 全量档 1 个约 1 分钟)")
    else:
        print("[警告] CUDA 不可用 —— 当前 torch 是 CPU 版, 训练会非常慢(要几天)。")
        print("请安装 CUDA 版 torch (约 2.5GB 下载):")
        print("  python -m pip install --force-reinstall torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")
        print("装完重新运行本脚本, 看到 '[OK] CUDA 可用' 再继续。")


if __name__ == "__main__":
    main()
