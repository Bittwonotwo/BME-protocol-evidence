# -*- coding: utf-8 -*-
"""统一引导模块 —— 所有入口脚本第一件事就 import 它。

它做两件事：
1) 设好 KMP_DUPLICATE_LIB_OK，规避 Anaconda 里 MKL 与 torch 的 libomp 冲突
   （否则会抛 "OMP: Error #15" 直接中止进程，且退出码还是 0，很容易误判为跑成功）。
2) 保证跑在「带 CUDA 的 python」上：如果当前解释器的 torch 用不了 GPU，
   自动找本机已装的 CUDA 解释器，用它重新执行同一个脚本。
   —— 这样你双击、或者用系统 python 敲命令，都不会莫名其妙跑到 CPU 上。

想指定别的解释器：设环境变量 MVP_PYTHON 即可。
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# AutoDL 等 Linux 镜像有时把 OMP_NUM_THREADS 设成非法值（空串、小数等），
# libgomp 每次启动都会刷 "Invalid value for environment variable OMP_NUM_THREADS"。
# 这里在任何 import torch 之前把非法值清掉（必须在加载 libgomp 之前做才有效）。
_v = os.environ.get("OMP_NUM_THREADS")
if _v is not None and not _v.isdigit():
    os.environ.pop("OMP_NUM_THREADS", None)

# 父进程要打印阶段标题、子进程直接写控制台，如果不实时刷新，父进程的话会迟到（看起来像顺序错乱）
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# 本机已知的 CUDA 环境（按优先级）。找不到时会退回当前解释器。
# 注意：跨解释器自动切换只在 Windows 本机有意义；Linux 服务器（如 AutoDL）上
# 当前解释器就是最终解释器，没有 CUDA 时直接报错并给 Linux 的修法。
if os.name == "nt":
    KNOWN_PYTHONS = [
        r"E:\ANACONDA\envs\condapythonproject\python.exe",
    ]
else:
    KNOWN_PYTHONS = []

_RELAUNCH_FLAG = "MVP_BOOTSTRAPPED"   # 防重入：已经换过一次就不再来


def _cuda_ok(py):
    """探测某个解释器能否用 CUDA（返回 True/False）"""
    try:
        r = subprocess.run(
            [py, "-c", "import torch;print(1 if torch.cuda.is_available() else 0)"],
            capture_output=True, text=True, timeout=180,
        )
        return r.stdout.strip().endswith("1")
    except Exception:
        return False


def _device_name(py):
    try:
        r = subprocess.run(
            [py, "-c", "import torch;print(torch.cuda.get_device_name(0))"],
            capture_output=True, text=True, timeout=180,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def ensure_cuda(require=True, quiet=False):
    """确保当前进程跑在带 CUDA 的 python 上。

    返回 True  = CUDA 可用（可能已切换到另一个解释器后继续执行）
    返回 False = 找不到可用的 CUDA（require=True 时直接退出并给修复建议）
    """
    from_torch = None
    try:
        import torch
        from_torch = torch
    except ImportError:
        pass

    if from_torch is not None and from_torch.cuda.is_available():
        return True

    # 已经换过一次还是不行，就别再循环了
    if not os.environ.get(_RELAUNCH_FLAG):
        cur = Path(sys.executable).resolve()
        cands = [os.environ.get("MVP_PYTHON", "")] + KNOWN_PYTHONS
        for c in cands:
            if not c:
                continue
            p = Path(c)
            if not p.exists() or p.resolve() == cur:
                continue
            if not _cuda_ok(str(p)):
                continue
            print(f"[引导] 当前解释器没有可用 CUDA: {sys.executable}")
            print(f"[引导] 自动切换到: {p}  ({_device_name(str(p))})")
            print(f"[引导] 重新执行: {Path(sys.argv[0]).name} {' '.join(sys.argv[1:])}\n")
            sys.stdout.flush()
            env = dict(os.environ)
            env[_RELAUNCH_FLAG] = "1"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            script = str(Path(sys.argv[0]).resolve())
            try:
                rc = subprocess.call([str(p), script, *sys.argv[1:]], env=env)
            except KeyboardInterrupt:
                rc = 130
            sys.exit(rc)

    if os.name == "nt":
        hint = (KNOWN_PYTHONS[0] if KNOWN_PYTHONS else "(未配置)")
        msg = ("[失败] 找不到能用的 CUDA 解释器。\n"
               "  当前解释器: {}\n"
               "  期望的解释器: {}\n"
               "  修法一: 用完整路径运行, 例如\n"
               "      \"{}\" run.py <步骤>\n"
               "  修法二: 先激活环境再运行\n"
               "      conda activate condapythonproject\n"
               "      python run.py <步骤>\n"
               "  修法三: 装 CUDA 版 torch\n"
               "      python -m pip install --force-reinstall torch torchvision "
               "--index-url https://download.pytorch.org/whl/cu121\n"
               ).format(sys.executable, hint, hint)
    else:
        msg = ("[失败] 当前解释器不能用 CUDA: {}\n"
               "  Linux 服务器上请确认:\n"
               "  1) 已选对 conda 环境: conda activate <你的环境名>\n"
               "  2) 该环境里装的是 CUDA 版 torch: python -c \"import torch;print(torch.cuda.is_available())\"\n"
               "     若输出 False, 重装: pip install --force-reinstall torch torchvision\n"
               "  3) 服务器有 GPU 且没被别的任务占满: nvidia-smi\n"
               ).format(sys.executable)
    if require:
        raise SystemExit(msg)
    if not quiet:
        print(msg)
    return False
