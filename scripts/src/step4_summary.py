# -*- coding: utf-8 -*-
"""步骤4: 汇总 —— 读取 输出/runs_mvp 下所有 run 的结果, 生成对比表与自动判定。

运行: python src/step4_summary.py    (或 python run.py summary)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def main():
    runs = ROOT / "输出" / "runs_mvp"
    n = len(list(runs.glob("*/final_metrics.json"))) if runs.exists() else 0
    print(f"检测到 {n} 个已完成的 run (快速档应为 8 个, 标准档 16 个)")
    if n == 0:
        raise SystemExit("[失败] 还没有任何 run 结果, 请先运行: python run.py formal")

    subprocess.run(
        [sys.executable, str(SRC / "summarize.py"),
         "--project", str(runs),
         "--out-prefix", str(ROOT / "输出" / "汇总结果")],
        cwd=str(ROOT), check=True,
    )
    print("\n" + "=" * 60)
    print("汇总已生成: 输出\\汇总结果.md")
    print("请把这个 md 文件的内容全文发回对话, 我按判定标准出结论。")
    print("建议同时附上: 输出\\runs_mvp\\g0_shot10_s1\\results.png")
    print("              输出\\runs_mvp\\g2_shot10_s1\\results.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
