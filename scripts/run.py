# -*- coding: utf-8 -*-
"""小目标 MVP 实验包 —— 统一入口（纯 Python，不需要 .bat）

用法（在本文件夹下打开终端）：

    python run.py              交互菜单，按数字选步骤
    python run.py check        1) 环境检查：看 CUDA / 显卡 / 显存 / batch 建议
    python run.py smoke        2) 冒烟测试：假数据跑 4 个小实验，3-5 分钟
    python run.py formal       3) 正式实验：读 配置.txt，跑 8 或 16 个训练
    python run.py summary      4) 汇总结果：生成 输出/汇总结果.md
    python run.py all          依次执行 check -> formal -> summary

说明：
  * 所有步骤都可以直接用 `python 某脚本.py` 单独运行（脚本在 src/ 里）。
  * 无论你用哪个 python 启动，脚本都会自动切换到带 CUDA 的解释器再执行
    （见 src/_bootstrap.py），不会静默跑到 CPU 上。
  * 想强制指定解释器：先 set MVP_PYTHON=D:\\path\\to\\python.exe
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import _bootstrap  # noqa: E402  (必须早于任何 torch 相关导入)

# 步骤定义: key -> (标题, 入口脚本, 是否强制要求 CUDA)
STEPS = {
    "check":   ("1) 环境检查 (CUDA / 显存 / batch 建议)", SRC / "check_env.py", False),
    "smoke":   ("2) 冒烟测试 (假数据, 3-5 分钟)",          SRC / "smoke.py",     False),
    "formal":  ("3) 正式实验 (读 配置.txt, 8/16 个训练)",   SRC / "run_formal.py", True),
    "summary": ("4) 汇总结果 (生成 输出/汇总结果.md)",      SRC / "step4_summary.py", False),
    "grid":    ("5) 网格实验 (头宽度 x 标注量, 40 个训练 ≈ 1.5h)", SRC / "run_grid.py", True),
    "gridsum": ("6) 网格汇总 (生成 输出/网格结果.md)",      SRC / "summarize_grid.py", False),
}
ORDER = ["check", "smoke", "formal", "summary", "grid", "gridsum"]


def run_step(key):
    title, script, need_cuda = STEPS[key]
    if not script.exists():
        raise SystemExit(f"[失败] 找不到 {script}")
    # 先确认解释器（必要时会切换并重新执行本脚本，切换前的输出会重复一次，属正常）
    _bootstrap.ensure_cuda(require=need_cuda)
    print("\n" + "=" * 66)
    print(f" {title}")
    print("=" * 66)
    rc = subprocess.call([sys.executable, str(script)], cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"\n[失败] 步骤 {key} 退出码 {rc}，已中止。修复后重跑本命令即可从断点继续。")
    return rc


def menu():
    print("小目标 MVP 实验包 —— 请选择步骤：\n")
    for k in ORDER:
        print(f"  {k:<9} {STEPS[k][0]}")
    print(f"  {'all':<9} 依次执行 check -> formal -> summary")
    print()
    try:
        val = input("输入步骤名（直接回车 = check）: ").strip().lower()
    except EOFError:
        val = "check"
    return val or "check"


def main():
    ap = argparse.ArgumentParser(add_help=True, description="小目标 MVP 实验包统一入口")
    ap.add_argument("step", nargs="?", default=None,
                    help="check / smoke / formal / summary / all，留空进入交互菜单")
    args = ap.parse_args()

    # 先在菜单出现之前把解释器切好，避免切换后整段菜单重复打印
    _bootstrap.ensure_cuda(require=False, quiet=True)

    step = args.step or menu()
    if step not in STEPS and step != "all":
        raise SystemExit(f"[失败] 未知步骤 '{step}'，可选: {', '.join(ORDER)} 或 all")

    if step == "all":
        run_step("check")
        run_step("formal")
        run_step("summary")
    else:
        run_step(step)
    print()


if __name__ == "__main__":
    main()
