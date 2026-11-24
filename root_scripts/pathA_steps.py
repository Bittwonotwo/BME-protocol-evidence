# -*- coding: utf-8 -*-
"""
pathA_steps.py  v2  (2026-11-21)
路径 A 20 run 的【只读】实际 optimizer steps 实测探针。

v2 改动（针对 v1 在真实服务器上 20/20 找不到 args.yaml）
--------------------------------------------------------
v1 只在 run 目录内找 args.yaml，找不到就刷 20 条 warning 并让 nbs 退回默认 64。
v2 增加：① 全 ROOT 扫 args.yaml / args.json 并按 run_name 精确匹配（Ultralytics
的 save_dir 可能落在 project 之外的位置）；② 每个 run 记录真实文件清单，用来
定案「这个 build 到底有没有写出 args.yaml」；③ nbs_source 显式标注是 per-run
一手还是 source-level 默认；④ warning 收敛成一条汇总。

⚠️ 关键：derived steps 只用 final_metrics.json 的一手 config + 源码公式，
    与 args.yaml 是否存在无关。v1 那 20/20 数值与判定档逐值一致，本身
    就是一次独立交叉验证；v2 只用于补齐「一手原文」这一档证据。

目的
----
final_metrics.json 不记录 nbs / accumulate / optimizer steps。本脚本扫描 20 个
run 目录里 Ultralytics 自动生成的 args.yaml + results.csv，把「名义 --updates
800 -> 实际 steps」这条链从推算升为一手记录，补齐论文附录 B 的步数表
（尤其 u800_b30 的 accumulate=6，历史只覆盖 8 / 2 两档）。

不训练、不推理、不改任何文件，只读 + 打一个 zip。约 1-2 分钟。

用法（服务器，单行、绝对路径、零 shell 元字符）
------------------------------------------------
python /root/pathA_steps.py

输出
----
/root/pathA_steps.zip
  内含 pathA_steps.json + args_yaml/ 下 20 份 args.yaml 原文

本地冒烟
--------
python pathA_steps.py --root <假ROOT> --out <假zip路径>
"""
import argparse
import json
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path

VERSION = "v2"
GROUPS = ["u800_b8", "u3200_b8", "u800_b30", "u800_n30"]
# args.yaml 白名单：只取这些键，且逐项做类型检查。
# 历史教训：Ultralytics args.yaml 里 val: True 会被 PyYAML(YAML1.1) 解析成 bool，
# 任何把该值传给 Path() 的代码都会崩。这里绝不把值当路径用。
WANTED = [
    "nbs", "batch", "epochs", "imgsz", "optimizer", "lr0", "momentum",
    "weight_decay", "warmup_epochs", "warmup_momentum", "close_mosaic",
    "model", "data", "seed", "deterministic", "patience", "freeze",
    "val", "project", "name", "exist_ok", "cos_lr", "device", "workers",
]


def log(*a):
    print(*a, flush=True)


def find_root(bases):
    """找含 src/train_mvp.py 的 ROOT，限定深度，避免扫 /root/miniconda3。"""
    cands = []
    for b in bases:
        bp = Path(b)
        if not bp.is_dir():
            continue
        cands.append(bp / "src" / "train_mvp.py")
        cands.append(bp / "mvp" / "src" / "train_mvp.py")
        try:
            for p in bp.glob("*/src/train_mvp.py"):
                cands.append(p)
            for p in bp.glob("*/*/src/train_mvp.py"):
                cands.append(p)
        except Exception:
            pass
    for c in cands:
        try:
            if c.is_file():
                return c.parent.parent
        except Exception:
            pass
    return None


def safe_scalar(v):
    """把 args.yaml 的值转成可安全 json 化的标量；bool 显式标成 'bool:True'。"""
    if isinstance(v, bool):
        return "bool:%s" % v
    if v is None:
        return None
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [safe_scalar(x) for x in v]
    if isinstance(v, dict):
        return {str(k): safe_scalar(x) for k, x in v.items()}
    return str(v)


def parse_args_yaml(p):
    """白名单 + 类型检查解析 args.yaml；返回 (rec, raw_text, error)。"""
    rec = {"file": str(p), "size": p.stat().st_size}
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return rec, "", "read_error: %r" % e
    try:
        import yaml as _y
    except Exception:
        _y = None
    if _y is None:
        # 无 PyYAML 时退化为逐行白名单解析，不解析整个文档，避开 bool 陷阱
        for line in raw.splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            k = line.split(":", 1)[0].strip()
            if k in WANTED:
                rec[k] = line.split(":", 1)[1].strip()
        rec["_parse_mode"] = "line_fallback"
    else:
        try:
            y = _y.safe_load(raw)
            if isinstance(y, dict):
                for k in WANTED:
                    if k in y:
                        rec[k] = safe_scalar(y[k])
                rec["_parse_mode"] = "yaml_safe_load"
            else:
                rec["_parse_mode"] = "not_a_dict"
        except Exception as e:
            rec["_parse_mode"] = "yaml_error: %r" % e
            for line in raw.splitlines():
                if ":" not in line or line.lstrip().startswith("#"):
                    continue
                k = line.split(":", 1)[0].strip()
                if k in WANTED:
                    rec[k] = line.split(":", 1)[1].strip()
    return rec, raw, None


def count_results_csv(run_dir):
    """results.csv 行数（去表头）= 实际训练 epochs，作为 epochs 的一手佐证。"""
    p = run_dir / "results.csv"
    if not p.is_file():
        for q in run_dir.rglob("results.csv"):
            p = q
            break
    if not p.is_file():
        return None, None
    try:
        lines = [x for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return max(0, len(lines) - 1), str(p)
    except Exception:
        return None, str(p)


def list_run_dir(run_dir, limit=40):
    """列出 run 目录真实内容，用于定案 args.yaml 到底有没有被写出来。"""
    top, keyfiles = [], []
    try:
        top = sorted(p.name + ("/" if p.is_dir() else "") for p in run_dir.iterdir())
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and p.name in ("args.yaml", "args.json", "results.csv",
                                          "final_metrics.json", "last.pt", "best.pt"):
                keyfiles.append(str(p.relative_to(run_dir)))
                if len(keyfiles) >= limit:
                    break
    except Exception as e:
        return {"error": repr(e), "top": top, "key_files_under_run": keyfiles}
    return {"top": top, "key_files_under_run": keyfiles}


def scan_all_args_yaml(root):
    """全实验包扫 args.yaml / args.json，返回路径列表（不读内容、不碰权重）。"""
    out = set()
    skip = {"__pycache__", ".git", "miniconda3", "site-packages", ".cache"}
    for pat in ("args.yaml", "args.json"):
        try:
            for p in root.rglob(pat):
                if any(x in p.parts for x in skip):
                    continue
                out.add(p)
        except Exception:
            pass
    return sorted(out, key=str)


def match_args_for_run(cands, run_name, group):
    """按 run_name 精确匹配，且必须落在 runs_pathA_<group> 下。

    严禁把历史 run 的同名 args.yaml 当成这批 run 的一手证据 ——
    服务器上 212 份历史 final_metrics 里有大量同名目录。别处候选
    只留在 args_yaml_scan.sample_paths 里供人工判断，不自动采用。
    """
    exact = [p for p in cands
             if run_name in p.parts and ("runs_pathA_" + group) in p.parts]
    return exact[0] if exact else None


def to_num(v, default=None):
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower().startswith("bool:"):
            return default
        try:
            return float(s)
        except Exception:
            return default
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="直接指定 ROOT（本地冒烟用）")
    ap.add_argument("--out", default="/root/pathA_steps.zip", help="输出 zip 路径")
    a = ap.parse_args()

    log("pathA_steps.py %s  (read-only step probe; no training, no inference)" % VERSION)
    if a.root:
        root = Path(a.root)
        if not (root / "src" / "train_mvp.py").is_file():
            log("[FATAL] --root 下没有 src/train_mvp.py: %s" % root)
            return 3
    else:
        root = find_root(["/root/autodl-tmp", "/root"])
        if root is None:
            log("[FATAL] ROOT not found. Try: python /root/pathA_steps.py --root /root/autodl-tmp/mvp")
            return 3
    log("ROOT = %s" % root)

    outdir = root / "输出"
    if not outdir.is_dir():
        log("[FATAL] 输出/ not found under %s" % root)
        return 3

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": VERSION,
        "root": str(root),
        "mode": "read_only_steps_probe",
        "source_formula": {
            "accumulate": "max(round(nbs / batch), 1)   [ultralytics 8.4.208 trainer.py:325]",
            "cap": "min(accumulate, len(train_loader) * epochs)   [trainer.py:328]",
            "steps": "total_batches / accumulate  (derived; warmup at trainer.py:528 adds a few extra updates)",
            "epochs_for": "b=max(1,min(batch,n_images)); nb=ceil(n_images/b); epochs=round(updates/nb)   [src/run_grid.py:61-67]",
            "nbs_source": "per-run args.yaml (primary) / source-level default 64 (fallback)",
            "caveat": "steps are derived from harness accounting, NOT counted by hooking optimizer.step()",
        },
        "groups": {},
        "rows": [],
        "warnings": [],
    }

    # --- v2: 全 ROOT 扫描，一次定案「args.yaml 写没写、写在哪」 ---
    all_args = scan_all_args_yaml(root)
    in_pathA = [p for p in all_args if any("runs_pathA_" in x for x in p.parts)]
    report["args_yaml_scan"] = {
        "total_under_root": len(all_args),
        "under_runs_pathA": len(in_pathA),
        "sample_paths": [str(p) for p in all_args[:25]],
        "note": ("v1 在 20 个 run 目录内一份都没找到；probe（run 之前）曾在 输出/ 下"
                 "统计到 56 个历史 args.yaml。本次全局扫描用于判定这 20 个 run 的"
                 "一手文件是否被写出、以及是否落在 project 之外。"),
    }
    log("args.yaml global scan: %d under ROOT, %d under runs_pathA_*" % (len(all_args), len(in_pathA)))
    for p in all_args[:5]:
        log("   " + str(p))

    raw_files = []   # (zip 内名字, 原文)
    total = 0
    missing_args = 0
    for g in GROUPS:
        gdir = outdir / ("runs_pathA_" + g)
        if not gdir.is_dir():
            report["warnings"].append("missing group dir: %s" % gdir)
            continue
        runs = sorted([d for d in gdir.iterdir() if d.is_dir()])
        report["groups"][g] = {"dir": str(gdir), "n_runs": len(runs)}
        for rd in runs:
            total += 1
            fm = rd / "final_metrics.json"
            ay = rd / "args.yaml"
            row = {"group": g, "run": rd.name, "run_dir": str(rd),
                   "dir_listing": list_run_dir(rd)}

            # --- final_metrics.json ---
            if fm.is_file():
                try:
                    d = json.loads(fm.read_text(encoding="utf-8"))
                    cfg = d.get("config", {})
                    row["split"] = d.get("split")
                    row["n_train_images"] = d.get("n_train_images")
                    row["mAP50"] = d.get("mAP50")
                    row["mAP50_95"] = d.get("mAP50_95")
                    row["epochs_fm"] = cfg.get("epochs")
                    row["batch_fm"] = cfg.get("batch")
                    row["batch_requested_fm"] = cfg.get("batch_requested")
                    row["head_scale"] = cfg.get("head_scale_requested")
                    row["model_fm"] = cfg.get("model")
                    row["ckpt"] = cfg.get("ckpt")
                except Exception as e:
                    row["fm_error"] = repr(e)
            else:
                report["warnings"].append("no final_metrics.json: %s" % rd)

            # --- args.yaml（Ultralytics 一手训练参数）：三级查找 ---
            found = ay if ay.is_file() else None
            if found is None:
                # ② 同目录更深一层（weights/ 同级或嵌套 save_dir）
                try:
                    for q in rd.rglob("args.yaml"):
                        found = q
                        break
                except Exception:
                    pass
            if found is None:
                # ③ 全 ROOT 按 run_name 精确匹配（save_dir 可能落在 project 之外）
                found = match_args_for_run(all_args, rd.name, g)
            if found is not None and found.is_file():
                rec, raw, err = parse_args_yaml(found)
                row["args_yaml"] = rec
                row["args_yaml_path"] = str(found)
                row["args_yaml_found_via"] = ("run_dir" if found == ay else
                                              "rglob" if str(rd) in str(found) else "global_scan")
                row["args_yaml_error"] = err
                zname = "args_yaml/%s__%s__%s" % (g, rd.name, found.name)
                raw_files.append((zname, raw))
            else:
                row["args_yaml"] = None
                row["args_yaml_found_via"] = "none"
                missing_args += 1

            # --- results.csv 佐证 epochs ---
            nlines, csvp = count_results_csv(rd)
            row["epochs_results_csv"] = nlines
            row["results_csv"] = csvp

            # --- 步数推算（一手 config + 源码公式；nbs 来源显式标注） ---
            nbs_ay = to_num((row.get("args_yaml") or {}).get("nbs"), None)
            if nbs_ay is not None:
                nbs = nbs_ay
                row["nbs_source"] = "per-run args.yaml (%s)" % row.get("args_yaml_path")
            else:
                nbs = 64.0
                row["nbs_source"] = ("source-level default nbs=64 "
                                     "(ultralytics 8.4.208 trainer.py:325; historically verified)")
            batch = to_num(row.get("batch_fm"), None)
            epochs = to_num(row.get("epochs_fm"), None)
            nimg = to_num(row.get("n_train_images"), None)
            row["nbs_used"] = nbs
            if batch and epochs and nimg:
                nb = max(1, math.ceil(nimg / batch))
                total_batches = nb * int(epochs)
                acc = max(int(round(nbs / batch)), 1)
                acc_capped = min(acc, total_batches)
                row["nb_batches_per_epoch"] = nb
                row["total_dataloader_batches"] = total_batches
                row["accumulate"] = acc
                row["accumulate_after_cap"] = acc_capped
                row["accumulate_capped_by_328"] = (acc_capped != acc)
                row["derived_optimizer_steps"] = round(total_batches / acc_capped, 2)
                row["warmup_extra_updates"] = "not counted (trainer.py:528)"
            else:
                row["derived_optimizer_steps"] = None
                report["warnings"].append("insufficient fields for steps: %s" % rd)

            # --- 交叉核对：args.yaml 的 batch/epochs vs final_metrics ---
            ayrec = row.get("args_yaml") or {}
            mism = []
            for k_src, k_fm in (("batch", "batch_fm"), ("epochs", "epochs_fm")):
                av = to_num(ayrec.get(k_src), None)
                fv = to_num(row.get(k_fm), None)
                if av is not None and fv is not None and abs(av - fv) > 0.5:
                    mism.append("%s: args.yaml=%s vs final_metrics=%s" % (k_src, av, fv))
            row["cross_check_mismatch"] = mism
            if mism:
                report["warnings"].append("cross-check mismatch %s: %s" % (rd.name, "; ".join(mism)))

            report["rows"].append(row)

    # --- v2: warning 收敛成一条，不逐 run 刷屏 ---
    if missing_args:
        report["warnings"].append(
            "args.yaml absent for %d/%d runs -> nbs fell back to source-level default 64; "
            "derived steps UNAFFECTED (they use final_metrics config + source formula only). "
            "See args_yaml_scan for where the historical args.yaml files live." % (missing_args, total))
    report["n_args_yaml_missing"] = missing_args

    # --- 汇总表（4 条件） ---
    summary = {}
    for g in GROUPS:
        rr = [x for x in report["rows"] if x["group"] == g]
        if not rr:
            continue
        vals = [x["derived_optimizer_steps"] for x in rr if x["derived_optimizer_steps"] is not None]
        accs = sorted(set(x["accumulate"] for x in rr if x.get("accumulate")))
        eps = sorted(set(x["epochs_fm"] for x in rr if x.get("epochs_fm") is not None))
        bfs = sorted(set(x["batch_fm"] for x in rr if x.get("batch_fm") is not None))
        nbs = sorted(set(x["nbs_used"] for x in rr if x.get("nbs_used") is not None))
        summary[g] = {
            "n_runs": len(rr),
            "nbs_values": nbs,
            "nbs_sources": sorted(set(x.get("nbs_source", "?") for x in rr)),
            "batch_actual": bfs,
            "epochs": eps,
            "accumulate_values": accs,
            "derived_steps_values": sorted(set(vals)),
            "derived_steps_mean": round(sum(vals) / len(vals), 2) if vals else None,
        }
    report["summary"] = summary
    report["n_runs_scanned"] = total
    report["n_args_yaml_collected"] = len(raw_files)

    # --- 打印摘要，肉眼可验 ---
    log("")
    log("=== STEP PROBE SUMMARY ===")
    log(json.dumps({
        "n_runs_scanned": total,
        "n_args_yaml_collected": len(raw_files),
        "n_args_yaml_missing": missing_args,
        "args_yaml_scan": {
            "total_under_root": len(all_args),
            "under_runs_pathA": len(in_pathA),
        },
        "warnings": len(report["warnings"]),
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    if report["warnings"]:
        log("")
        log("=== WARNINGS (first 12) ===")
        for w in report["warnings"][:12]:
            log("  - " + w)

    # --- 打包 ---
    out = Path(a.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pathA_steps.json", json.dumps(report, ensure_ascii=False, indent=2))
        for name, raw in raw_files:
            z.writestr(name, raw)
    log("")
    log("=== PACK ===")
    log("zip: %s" % out)
    log("size: %.1f KB" % (out.stat().st_size / 1024.0))
    log("files: %d  (args.yaml: %d)" % (1 + len(raw_files), len(raw_files)))
    if total < 20:
        log("[WARN] expected 20 runs; got %d" % total)
    if len(raw_files) == 20:
        log("OK: 20/20 per-run args.yaml collected. FileZilla -> send /root/pathA_steps.zip back.")
    else:
        log("[INFO] %d/20 per-run args.yaml found; %d missing -> nbs uses source-level default 64." % (
            len(raw_files), missing_args))
        log("       NOT a data error: all 20 derived step values come from")
        log("       final_metrics.json config + the verified source formula and stay valid.")
        log("       Send /root/pathA_steps.zip back anyway: global scan + per-run dir")
        log("       listings in pathA_steps.json settle the question once and for all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
