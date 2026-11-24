# -*- coding: utf-8 -*-
"""步骤2: 冒烟测试 —— 用合成假数据跑通全链路, 验证你机器上的环境与模块改造是否正常。
只跑 4 个 run x 3 epoch。数值无意义, 只看是否全部成功。

运行: python src/smoke.py    (或 python run.py smoke)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "输出" / "冒烟"


def run(args):
    print("\n>>>", " ".join(str(a) for a in args[-6:]))
    r = subprocess.run([sys.executable, *[str(a) for a in args]], cwd=str(ROOT))
    if r.returncode != 0:
        print("\n[失败] 上一步出错, 冒烟测试中止。请把上面的报错发回对话。")
        sys.exit(1)


def main():
    _bootstrap.ensure_cuda(require=False)   # 没 GPU 也能跑通链路, 只是慢
    OUT.mkdir(parents=True, exist_ok=True)
    run([SRC / "make_dummy_data.py", "--out", OUT / "dummy_src"])
    run([SRC / "prepare_data.py", "--src", OUT / "dummy_src", "--out", OUT / "data",
         "--names", "defect_a,defect_b"])
    run([SRC / "run_all.py", "--data-root", OUT / "data", "--project", OUT / "runs",
         "--splits", "shot10_s1", "--epochs", "3"])
    run([SRC / "summarize.py", "--project", OUT / "runs", "--out-prefix", OUT / "冒烟汇总"])

    print("\n" + "=" * 60)
    print("冒烟测试通过: 环境正常, 四个组的模块改造都能跑通。")
    print("提示: 上面的数字是假数据 + 3 个 epoch, 没有参考价值, 不用管它。")
    print("下一步: python run.py formal")
    print("=" * 60)


if __name__ == "__main__":
    main()
