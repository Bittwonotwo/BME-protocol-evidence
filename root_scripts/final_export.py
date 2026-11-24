# -*- coding: utf-8 -*-
"""
final_export.py  v1  (2026-11-21)
==================================
最终撤离打包：把服务器上全部实验证据一次性带回本地电脑。

【只读】不训练、不推理、不修改/删除服务器上的任何文件。
【自包含】不依赖 tfa_selfcheck.py 等任何其它脚本，只传这一个文件即可跑。

背景（为什么需要这个脚本）：
  之前的 evidence_pack.zip 有两个缺口：
    (1) 它是在路径 A 20 run 之前打的，只含 212 份 final_metrics；
    (2) archive_evidence.py 的 SKIP_DIR 含 'runs'，把 Ultralytics 写到
        ROOT/runs/detect/输出/<project>/<name>/args.yaml 的一手训练参数全跳过了。
  pathA_steps.py v2 全局扫描证实：ROOT 下共 232 份 args.yaml
  （212 历史 + 20 路径 A），与 232 个 run 一一对应。
  本脚本把这两块补齐，并生成可直接进开源仓库 / 论文附录的总索引。

用法（服务器 /root 下，全部单行、绝对路径、零 shell 元字符）：

  python /root/final_export.py
      -> /root/final_export.zip        轻量证据总包（推荐，不含权重与图像）

  python /root/final_export.py --weights runs_tfa
      -> /root/final_weights_runs_tfa.zip   只带某一组的 .pt（可换 runs_b8a /
         runs_pathA_u800_b8 等组名；--weights all 为全部，约 9 GB，慎用）

  python /root/final_export.py --dataset
      -> /root/final_dataset.zip        划分数据集（图像 + 标签 + yaml）

可选：--root /root/autodl-tmp/mvp  显式指定 ROOT
      --out /root/final_export.zip    显式指定输出

zip 内保留服务器相对 ROOT 的完整目录层级（镜像树），不会同名覆盖。
默认包含：
  final_metrics/  232 份 final_metrics.json（212 主体 + 20 路径 A）
  args_yaml/      232 份 Ultralytics 一手 args.yaml
  results_csv/    232 份训练曲线 results.csv
  selfcheck/      tfa 两次自查 + pathA 探针 / probe / meta 的 json
  vis/            两次自查的叠放对比图
  data_yaml/      划分 yaml（mvp_data 27 + mvp_data_neu 17）
  data_labels/    划分标签 txt
  scripts/        src/*.py 与顶层 *.py（含本次所有探针脚本）
  docs/           顶层 md / 配置.txt
  run_index.csv   232 run 汇总索引（原始字段 + 明确标注的 derived 步数）
  weights_index.csv  全部 .pt 清单（路径 + 体积，不含本体）
  manifest.json   计数、校验、公式与 caveat
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

VERSION = "v1"

# 只跳过纯缓存目录；绝对不能跳 'runs' —— 232 份 args.yaml 就在 runs/detect/ 下
SKIP_DIR = {"__pycache__", ".git", ".cache", ".pytest_cache", "node_modules"}
SKIP_SUFFIX = {".pt", ".pth", ".pkl", ".pyc", ".onnx", ".engine", ".npy", ".npz"}
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

EXPECTED_RUNS = 232
DEFAULT_NBS = 64.0            # 源码级一手（ultralytics 8.4.208 trainer.py:325）
MAX_SINGLE_CSV = 3 * 1024 * 1024   # 单份 results.csv 超过 3 MB 则跳过并告警

SELFCHECK_NAMES = [
    "tfa_scan_report.json",
    "tfa_selfcheck_result.json",
    "tfa_stage2_result.json",
    "pathA_steps.json",
    "pathA_probe.json",
    "pathA_meta.json",
    "evidence_manifest.json",
]

ROOT_CANDIDATES = [
    "/root/autodl-tmp/mvp",
    "/root/mvp",
    "/root/autodl-tmp",
    "/root",
]


def log(*a):
    print(*a, flush=True)


def find_root(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if (p / "src" / "train_mvp.py").is_file() else None
    for c in ROOT_CANDIDATES:
        p = Path(c)
        if (p / "src" / "train_mvp.py").is_file():
            return p
    # 兜底：受控 BFS，最多 4 层，剪掉缓存目录
    for base in ["/root/autodl-tmp", "/root"]:
        bp = Path(base)
        if not bp.is_dir():
            continue
        frontier = [bp]
        for _ in range(4):
            nxt = []
            for d in frontier:
                if (d / "src" / "train_mvp.py").is_file():
                    return d
                try:
                    for sub in d.iterdir():
                        if sub.is_dir() and sub.name not in SKIP_DIR:
                            nxt.append(sub)
                except Exception:
                    pass
            frontier = nxt
    return None


def scan_exact(root, filename):
    """按文件名精确扫描，保留完整目录层级。剪枝只跳缓存，不跳 runs。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        if filename in filenames:
            out.append(Path(dirpath) / filename)
    return sorted(out)


def scan_suffix(root, suffixes):
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        for fn in filenames:
            if Path(fn).suffix.lower() in suffixes:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def rel(p, root):
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def group_of(relpath):
    """从 '输出/runs_xxx/<run>/final_metrics.json' 里取出 runs_xxx。

    注意：args.yaml 落在 'runs/detect/输出/runs_xxx/<run>/args.yaml'，
    前面的 'runs' / 'detect' 不是组名，必须优先取 runs_ 前缀的最后一段。
    """
    parts = relpath.split("/")
    cands = [x for x in parts if x.startswith("runs_")]
    if cands:
        return cands[-1]
    for x in parts:
        if x.startswith("runs") and x not in ("runs", "detect"):
            return x
    return parts[-2] if len(parts) >= 2 else "?"


def run_of(relpath):
    parts = relpath.split("/")
    return parts[-2] if len(parts) >= 2 else "?"


def safe_num(v, default=None):
    if isinstance(v, bool) or v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return default


def load_steps_index(root):
    """从 pathA_steps.json 读 20 run 的一手 nbs / accumulate / steps；找不到返回 {}。"""
    idx = {}
    for p in scan_exact(root, "pathA_steps.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("rows", []):
            g_raw = str(r.get("group", ""))
            rn = str(r.get("run", ""))
            rec = {
                "nbs": safe_num(r.get("nbs_used"), None),
                "accumulate": safe_num(r.get("accumulate"), None),
                "steps": safe_num(r.get("derived_optimizer_steps"), None),
                "args_yaml_path": r.get("args_yaml_path", ""),
                "source": "per-run args.yaml + harness accounting",
            }
            # pathA_steps.json 里 group 是 'u800_b8'，而 fm 目录是
            # '输出/runs_pathA_u800_b8/' —— 两个 key 都存，确保命中。
            idx[(g_raw, rn)] = rec
            idx[("runs_pathA_" + g_raw, rn)] = rec
    return idx


def derived_from_fm(cfg, n_img):
    """按已定案源码公式从 final_metrics 推步数（用于 212 历史 run）。"""
    ep = safe_num(cfg.get("epochs"), None)
    b = safe_num(cfg.get("batch"), None)
    n = safe_num(n_img, None)
    if not ep or not b or not n:
        return None, None, None
    nb = max(1, int(math.ceil(n / b)))
    acc = max(int(round(DEFAULT_NBS / b)), 1)
    steps = ep * nb / float(acc)
    return DEFAULT_NBS, acc, round(steps, 2)


def build_run_index(root, fm_files, args_files, steps_idx):
    args_rel = set(rel(p, root) for p in args_files)
    rows = []
    for p in fm_files:
        rp = rel(p, root)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"fm_relpath": rp, "parse_error": repr(e)})
            continue
        cfg = d.get("config", {}) or {}
        n_img = d.get("n_train_images")
        g = group_of(rp)
        rn = run_of(rp)
        rec = {
            "fm_relpath": rp,
            "group": g,
            "run": rn,
            "split": d.get("split", ""),
            "n_train_images": n_img,
            "epochs": cfg.get("epochs"),
            "batch": cfg.get("batch"),
            "batch_requested": cfg.get("batch_requested"),
            "head_scale_requested": cfg.get("head_scale_requested"),
            "model": cfg.get("model"),
            "ckpt": cfg.get("ckpt"),
            "mAP50": d.get("mAP50"),
            "mAP50_95": d.get("mAP50_95"),
            "freeze": cfg.get("freeze"),
            "oom_downgraded": cfg.get("oom_downgraded"),
            "has_args_yaml": "",
            "nbs_used": "",
            "accumulate": "",
            "derived_optimizer_steps": "",
            "steps_source": "",
        }
        # 该 run 的 args.yaml 是否存在（镜像到 runs/detect/ 下）
        cand = [a for a in args_rel if rn in a.split("/") and g in a]
        rec["has_args_yaml"] = "yes" if cand else "no"
        hit = steps_idx.get((g, rn))
        if hit:
            rec["nbs_used"] = hit["nbs"]
            rec["accumulate"] = hit["accumulate"]
            rec["derived_optimizer_steps"] = hit["steps"]
            rec["steps_source"] = hit["source"]
        else:
            nbs, acc, steps = derived_from_fm(cfg, n_img)
            rec["nbs_used"] = nbs if nbs is not None else ""
            rec["accumulate"] = acc if acc is not None else ""
            rec["derived_optimizer_steps"] = steps if steps is not None else ""
            rec["steps_source"] = ("source default nbs=64 + harness accounting"
                                   if steps is not None else "insufficient fields")
        rows.append(rec)
    return rows


def collect(root, home):
    g = {}
    g["final_metrics"] = scan_exact(root, "final_metrics.json")
    g["args_yaml"] = scan_exact(root, "args.yaml")
    g["results_csv"] = [p for p in scan_exact(root, "results.csv")
                        if p.stat().st_size <= MAX_SINGLE_CSV]

    # selfcheck / probe / meta：按文件名全局找
    sc = []
    for n in SELFCHECK_NAMES:
        sc += scan_exact(root, n)
    # 去重
    seen, uniq = set(), []
    for p in sorted(sc):
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    g["selfcheck"] = uniq

    # 叠放图：只取自查 vis 目录，不取 Ultralytics 训练图（体积大）
    vis = []
    for d in ["tfa_selfcheck_vis", "tfa_stage2_vis"]:
        v = root / "输出" / d
        if v.is_dir():
            vis += scan_suffix(v, {".png"})
    g["vis"] = sorted(set(vis))

    # 划分 yaml + 标签（不含图像）
    dy, dl = [], []
    for sub in ["mvp_data", "mvp_data_neu"]:
        base = root / "输出" / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            s = p.suffix.lower()
            if s in (".yaml", ".yml"):
                dy.append(p)
            elif s == ".txt":
                dl.append(p)
    g["data_yaml"] = sorted(set(dy))
    g["data_labels"] = sorted(set(dl))

    # 脚本 + 文档
    scripts = []
    for pat in ["src/*.py", "*.py"]:
        for p in sorted(root.glob(pat)):
            if p.suffix.lower() in SKIP_SUFFIX:
                continue
            if any(part in SKIP_DIR for part in p.parts):
                continue
            if p not in scripts:
                scripts.append(p)
    g["scripts"] = scripts
    g["docs"] = [p for p in sorted(root.glob("*.md"))] + \
                [p for p in sorted(root.glob("*.txt"))]

    # /root 顶层的脚本（服务器上跑的那一版，用来核对版本）与产出清单
    root_scripts, root_artifacts = [], []
    if home.is_dir():
        for p in sorted(home.glob("*.py")):
            if p.is_file() and p.suffix.lower() not in SKIP_SUFFIX:
                root_scripts.append(p)
        for pat in ("*.zip", "*.json", "*.md", "*.txt"):
            for p in sorted(home.glob(pat)):
                if p.is_file():
                    try:
                        sz = p.stat().st_size
                    except Exception:
                        sz = 0
                    root_artifacts.append({"name": p.name, "size_bytes": sz,
                                           "size_KB": round(sz / 1024.0, 1)})
    g["root_scripts"] = root_scripts
    return g, root_artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="显式指定 ROOT（含 src/train_mvp.py）")
    ap.add_argument("--root-home", default="/root",
                    help="顶层脚本所在目录（服务器就是 /root）")
    ap.add_argument("--out", default="/root/final_export.zip")
    ap.add_argument("--weights", default=None,
                    help="额外打包权重：组名（如 runs_tfa）或 all")
    ap.add_argument("--dataset", action="store_true",
                    help="额外打包划分数据集（含图像与标签）")
    a = ap.parse_args()

    log("final_export.py %s  (read-only final archive)" % VERSION)
    root = find_root(a.root)
    if root is None:
        log("[FATAL] 找不到 ROOT（需含 src/train_mvp.py）。可用 --root 指定。")
        return 3
    log("ROOT = %s" % root)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import ultralytics
        ver = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        ver = "not-imported"

    home = Path(a.root_home)
    g, root_artifacts = collect(root, home)
    for k in ["final_metrics", "args_yaml", "results_csv", "selfcheck",
              "vis", "data_yaml", "data_labels", "scripts", "docs",
              "root_scripts"]:
        log("[OK] %-16s %d" % (k, len(g[k])))
    log("[OK] %-16s %d" % ("root_artifacts", len(root_artifacts)))

    # 索引
    steps_idx = load_steps_index(root)
    n_step_runs = len(set(k[1] for k in steps_idx))
    log("[OK] pathA_steps.json 覆盖 %d 个 run（双 key 索引 %d 条）"
        % (n_step_runs, len(steps_idx)))
    ri = build_run_index(root, g["final_metrics"], g["args_yaml"], steps_idx)

    pts = scan_suffix(root, {".pt", ".pth"})
    wi = []
    for p in pts:
        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0
        wi.append({"relpath": rel(p, root), "group": group_of(rel(p, root)),
                   "size_bytes": sz, "size_MB": round(sz / 1048576.0, 2)})
    total_pt_bytes = sum(x["size_bytes"] for x in wi)
    log("[OK] .pt 清单 %d 个 / %.2f GB（不打包本体）"
        % (len(wi), total_pt_bytes / 1073741824.0))

    warnings = []
    if len(g["final_metrics"]) != EXPECTED_RUNS:
        warnings.append("final_metrics %d 份，期望 %d 份"
                        % (len(g["final_metrics"]), EXPECTED_RUNS))
    if len(g["args_yaml"]) < len(g["final_metrics"]):
        warnings.append("args.yaml %d 份，少于 final_metrics %d 份"
                        % (len(g["args_yaml"]), len(g["final_metrics"])))
    if len(g["results_csv"]) < len(g["final_metrics"]):
        warnings.append("results.csv %d 份（少于 run 数，部分 run 可能未写出或超 3 MB）"
                        % len(g["results_csv"]))
    for n in ["pathA_steps.json", "tfa_stage2_result.json"]:
        if not any(p.name == n for p in g["selfcheck"]):
            warnings.append("未找到 %s（可选，不影响打包）" % n)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script_version": VERSION,
        "root": str(root),
        "ultralytics": ver,
        "expected_final_metrics": EXPECTED_RUNS,
        "counts": {k: len(v) for k, v in g.items()},
        "n_root_artifacts": len(root_artifacts),
        "n_run_index_rows": len(ri),
        "weights": {"count": len(wi),
                    "total_bytes": total_pt_bytes,
                    "total_GB": round(total_pt_bytes / 1073741824.0, 2),
                    "packed": False},
        "source_formula": {
            "accumulate": "max(round(nbs / batch), 1)  [ultralytics 8.4.208 trainer.py:325]",
            "epochs_for": "b=min(batch,n_images); nb=ceil(n_images/b); epochs=round(updates/nb)  [run_grid.py:61-67]",
            "derived_steps": "epochs * nb / accumulate  (harness accounting, NOT step-by-step counted)",
            "nbs": "per-run args.yaml for the 20 pathA runs; source default 64 for the 212 historical runs",
        },
        "notes": [
            "默认不含 .pt 权重与图像；--weights / --dataset 另出独立 zip。",
            "zip 内保留相对 ROOT 的完整目录层级（root_scripts 除外），天然避免同名覆盖。",
            "root_scripts/ 收的是 /root 顶层实际跑过的脚本版本；"
            " root_artifacts.csv 是 /root 下历史产出包的清单（只列名与体积，不重复打包本体）。",
            "args.yaml 来自 ROOT/runs/detect/输出/<project>/<name>/，与脚本写的"
            " ROOT/输出/<project>/<name>/final_metrics.json 是两条不同路径。",
            "run_index.csv 的 derived_optimizer_steps 一律为推算列，"
            "不得写成 per-run measured；steps_source 列标明每个值的来源。",
            "本脚本只读，不修改或删除服务器上的任何文件。",
        ],
        "warnings": warnings,
    }

    prefix = {
        "final_metrics": "final_metrics",
        "args_yaml": "args_yaml",
        "results_csv": "results_csv",
        "selfcheck": "selfcheck",
        "vis": "vis",
        "data_yaml": "data_yaml",
        "data_labels": "data_labels",
        "scripts": "scripts",
        "docs": "docs",
        "root_scripts": "root_scripts",
    }

    def arcname(p, key):
        if key == "root_scripts":
            return "root_scripts/" + p.name
        return prefix[key] + "/" + rel(p, root)

    # 生成 csv / json 到内存
    def csv_text(rows, fields):
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()

    ri_fields = ["fm_relpath", "group", "run", "split", "n_train_images",
                 "epochs", "batch", "batch_requested", "head_scale_requested",
                 "model", "ckpt", "mAP50", "mAP50_95", "freeze",
                 "oom_downgraded", "has_args_yaml", "nbs_used", "accumulate",
                 "derived_optimizer_steps", "steps_source", "parse_error"]
    wi_fields = ["relpath", "group", "size_bytes", "size_MB"]
    ra_fields = ["name", "size_bytes", "size_KB"]

    n_entries = 0
    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("run_index.csv", csv_text(ri, ri_fields))
        z.writestr("weights_index.csv", csv_text(wi, wi_fields))
        z.writestr("root_artifacts.csv", csv_text(root_artifacts, ra_fields))
        n_entries += 4
        for key in prefix:
            for p in g[key]:
                try:
                    z.write(str(p), arcname(p, key))
                    n_entries += 1
                except Exception as e:
                    warnings.append("skip %s: %r" % (rel(p, root), e))
    log("")
    log("[OK] zip -> %s  (%.1f KB, %d 个条目)"
        % (out, out.stat().st_size / 1024.0, n_entries))
    for w in warnings:
        log("[WARN] %s" % w)

    # ---- 可选：权重 ----
    if a.weights:
        grp = a.weights.strip()
        sel = wi if grp.lower() == "all" else \
              [x for x in wi if grp in x["relpath"]]
        if not sel:
            log("[WARN] --weights %s 没有匹配到任何 .pt，未生成权重包" % grp)
        else:
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in grp)
            wo = Path("/root") / ("final_weights_%s.zip" % safe)
            if a.out != "/root/final_export.zip":
                wo = out.parent / ("final_weights_%s.zip" % safe)
            tot = sum(x["size_bytes"] for x in sel)
            log("")
            log("[WEIGHTS] %d 个 .pt / %.2f GB -> %s"
                % (len(sel), tot / 1073741824.0, wo))
            if tot > 2 * 1073741824:
                log("[WARN] 超过 2 GB：请确认服务器剩余磁盘与 FileZilla 传输时间")
            with zipfile.ZipFile(str(wo), "w", zipfile.ZIP_STORED) as z:
                for x in sel:
                    p = root / x["relpath"]
                    try:
                        z.write(str(p), "weights/" + x["relpath"])
                    except Exception as e:
                        log("[WARN] skip %s: %r" % (x["relpath"], e))
            log("[OK] weights zip -> %s  (%.1f MB)"
                % (wo, wo.stat().st_size / 1048576.0))

    # ---- 可选：数据集 ----
    if a.dataset:
        ds = []
        for sub in ["mvp_data", "mvp_data_neu"]:
            base = root / "输出" / sub
            if base.is_dir():
                ds += [p for p in base.rglob("*") if p.is_file()]
        do = Path("/root/final_dataset.zip")
        if a.out != "/root/final_export.zip":
            do = out.parent / "final_dataset.zip"
        with zipfile.ZipFile(str(do), "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(ds):
                z.write(str(p), "dataset/" + rel(p, root))
        log("")
        log("[OK] dataset zip -> %s  (%d 个文件, %.1f MB)"
            % (do, len(ds), do.stat().st_size / 1048576.0))

    log("")
    log("=== DONE ===")
    log("FileZilla 回传：%s" % out)
    for k in prefix:
        log("       %-16s %d" % (k, len(g[k])))
    log("       run_index.csv       %d 行" % len(ri))
    log("       weights_index.csv   %d 行（清单，不含本体）" % len(wi))
    log("       root_artifacts.csv  %d 行（清单，不含本体）" % len(root_artifacts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
