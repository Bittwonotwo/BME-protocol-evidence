# -*- coding: utf-8 -*-
"""
Path A helper: budget-matched re-evaluation of the CDT condition reported in
Ghali, Benhafid & Selouani, "Benchmarking Few-Shot Learning Techniques for
Steel Surface Defect Detection", 2025 IEEE Smart World Congress (SWC),
pp. 9-14, DOI 10.1109/SWC65939.2025.00031  (code: github.com/Rayen023/SSL-YOLO).

Three modes, all single-line commands, absolute paths, no shell metacharacters:

    python /root/pathA.py --probe
    python /root/pathA.py --run
    python /root/pathA.py --pack

--probe : read-only inventory + validates the 4 planned run_grid.py calls
          via run_grid.py --dry-run. Writes /root/pathA_probe.zip
--run   : executes the 20-run re-evaluation. Writes /root/pathA_runs.zip
--pack  : repacks results only (use if --run was interrupted).

Nothing here installs packages, modifies source files, or deletes anything.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# ---- re-evaluation matrix: 4 conditions x 5 draws = 20 runs -----------------
# All use head-scale 1.0 (= unmodified detector head, i.e. the CDT condition:
# COCO-pretrained YOLO backbone + few-shot fine-tuning, no extra modules).
PLAN = [
    {"tag": "u800_b8",  "shots": [10], "updates": 800,  "batch": 8},
    {"tag": "u3200_b8", "shots": [10], "updates": 3200, "batch": 8},
    {"tag": "u800_b30", "shots": [10], "updates": 800,  "batch": 30},
    {"tag": "u800_n30", "shots": [30], "updates": 800,  "batch": 8},
]
DRAWS = 5
SCALES = [1.0]
PROJECT_PREFIX = "runs_pathA"
VERSION = "v2.1"  # bumped after fixing the Ultralytics args.yaml bool crash

SKIP_SUFFIX = {".pt", ".pth", ".pkl", ".pyc", ".onnx", ".engine", ".npy", ".npz"}
SKIP_DIR = {"__pycache__", ".git", ".cache", "weights"}


def log(msg):
    print(msg, flush=True)


def find_root(base):
    """Locate the experiment root by src/train_mvp.py, shallow-first."""
    base = Path(base)
    if not base.is_dir():
        return None
    hits = []
    for depth in range(0, 6):
        for p in base.glob("*/" * depth + "src/train_mvp.py"):
            hits.append(p)
            if len(hits) >= 8:
                break
        if hits:
            break
    if not hits:
        try:
            for p in base.rglob("src/train_mvp.py"):
                hits.append(p)
                if len(hits) >= 8:
                    break
        except OSError:
            pass
    roots = sorted({p.parent.parent.resolve() for p in hits},
                   key=lambda x: len(str(x)))
    return roots[0] if roots else None


def env_info():
    d = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
            d["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception as e:
        d["torch_error"] = repr(e)
    try:
        import ultralytics
        d["ultralytics"] = getattr(ultralytics, "__version__", "?")
    except Exception as e:
        d["ultralytics_error"] = repr(e)
    try:
        import yaml
        d["pyyaml"] = getattr(yaml, "__version__", "?")
    except Exception as e:
        d["pyyaml_error"] = repr(e)
    return d


def scan_yaml(root):
    """List DATA yamls under ROOT/输出; skip Ultralytics training artefacts.

    v1 crashed with ``TypeError: expected str, bytes or os.PathLike object,
    not bool`` because rglob("*.yaml") also picked up the ``args.yaml`` that
    Ultralytics writes into every run dir.  Those files contain ``val: True``
    (YAML 1.1 turns yes/no/on/off/true/false into bool), and Path(True) explodes.

    Two guards now: (1) skip training-artefact yamls entirely, (2) only ever
    pass string-ish values into Path().  Returns (yamls, skipped_report).
    """
    out = {}
    skipped = {"training_args_yamls": 0, "under_runs_dir": 0,
               "non_string_values": []}
    try:
        import yaml as _y
    except Exception:
        _y = None
    outdir = root / "输出"
    if not outdir.is_dir():
        return out, skipped
    for p in sorted(outdir.rglob("*.yaml")):
        relparts = p.relative_to(root).parts
        # (1) Ultralytics writes args.yaml in each run dir: not a data yaml.
        if p.name.lower() == "args.yaml":
            skipped["training_args_yamls"] += 1
            continue
        if any(part.startswith("runs") for part in relparts):
            skipped["under_runs_dir"] += 1
            continue
        rel = str(p.relative_to(root))
        rec = {"file": str(p), "size": p.stat().st_size}
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            rec["read_error"] = repr(e)
            out[rel] = rec
            continue
        if _y is not None:
            try:
                y = _y.safe_load(txt)
                if isinstance(y, dict):
                    for key in ("path", "train", "val", "test", "nc", "names"):
                        rec[key] = y.get(key)
            except Exception as e:
                rec["yaml_error"] = repr(e)
        else:
            for key in ("path", "train", "val", "test", "nc"):
                for line in txt.splitlines():
                    if line.strip().startswith(key + ":"):
                        rec[key] = line.split(":", 1)[1].strip()
        # (2) only string-ish values may be treated as paths
        base = p.parent
        pv = rec.get("path")
        if isinstance(pv, (str, os.PathLike)) and str(pv).strip():
            pp = Path(str(pv))
            if pp.is_absolute():
                base = pp
        elif pv is not None and not isinstance(pv, (str, os.PathLike)):
            rec["path_non_string"] = repr(pv)
            if len(skipped["non_string_values"]) < 20:
                skipped["non_string_values"].append("%s:path=%r" % (rel, pv))
        for split in ("train", "val", "test"):
            v = rec.get(split)
            if not v:
                continue
            if not isinstance(v, (str, os.PathLike)):
                rec[split + "_non_string"] = repr(v)
                if len(skipped["non_string_values"]) < 20:
                    skipped["non_string_values"].append(
                        "%s:%s=%r" % (rel, split, v))
                continue
            v = str(v)
            imgdir = Path(v) if Path(v).is_absolute() else base / v
            if not imgdir.is_dir():
                rec[split + "_img_dir_missing"] = str(imgdir)
                continue
            imgs = [x for x in imgdir.iterdir()
                    if x.is_file() and x.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            rec[split + "_images"] = len(imgs)
            lbl = Path(str(imgdir).replace("images", "labels"))
            if lbl.is_dir():
                rec[split + "_labels"] = len([x for x in lbl.iterdir() if x.suffix == ".txt"])
        out[rel] = rec
    return out, skipped


def dataset_readiness(yamls):
    """Which dataset dirs have a complete shot10_s1..s5 / shot30_s1..s5 split."""
    res = {}
    for rel, rec in yamls.items():
        parent = str(Path(rel).parent)
        res.setdefault(parent, {"yaml_count": 0, "shot10": [], "shot30": [], "path": rec.get("path")})
        res[parent]["yaml_count"] += 1
        stem = Path(rel).stem
        if stem.startswith("shot10_s"):
            res[parent]["shot10"].append(stem)
        elif stem.startswith("shot30_s"):
            res[parent]["shot30"].append(stem)
    for k, v in res.items():
        v["shot10_ok"] = all(("shot10_s%d" % i) in v["shot10"] for i in range(1, DRAWS + 1))
        v["shot30_ok"] = all(("shot30_s%d" % i) in v["shot30"] for i in range(1, DRAWS + 1))
        v["shot10"] = sorted(v["shot10"])[:12]
        v["shot30"] = sorted(v["shot30"])[:12]
    return res


def scan_weights(root):
    out = []
    for p in sorted(root.rglob("*.pt")):
        if any(part in ("__pycache__", ".git", ".cache") for part in p.parts):
            continue
        try:
            out.append({"path": str(p), "mb": round(p.stat().st_size / 1e6, 1)})
        except OSError:
            pass
        if len(out) >= 40:
            break
    return out


def scan_runs(root):
    outdir = root / "输出"
    res = {"dirs": [], "final_metrics": 0}
    if not outdir.is_dir():
        return res
    for d in sorted(outdir.glob("runs*")):
        if d.is_dir():
            n = len(list(d.rglob("final_metrics.json")))
            res["dirs"].append({"name": d.name, "final_metrics": n})
            res["final_metrics"] += n
    return res


def build_cmd(root, item, extra=None):
    """One run_grid.py invocation as an argv list (never a shell string)."""
    script = root / "src" / "run_grid.py"
    cmd = [sys.executable, str(script),
           "--stage", "kill",
           "--shots"] + [str(x) for x in item["shots"]] + [
           "--scales"] + [str(x) for x in SCALES] + [
           "--draws", str(DRAWS),
           "--updates", str(item["updates"]),
           "--batch", str(item["batch"]),
           "--skip-data",
           "--project", "输出/" + PROJECT_PREFIX + "_" + item["tag"]]
    if extra:
        cmd += list(extra)
    return cmd


def probe(root, out_json, args):
    yamls, skipped = scan_yaml(root)
    d = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "mode": "probe",
        "env": env_info(),
        "yamls": yamls,
        "scan_skipped": skipped,
        "weights": scan_weights(root),
        "runs": scan_runs(root),
        "plan": [],
        "errors": [],
    }
    d["datasets"] = dataset_readiness(d["yamls"])

    for item in PLAN:
        cmd = build_cmd(root, item)
        rec = {"tag": item["tag"], "cmd": cmd}
        try:
            r = subprocess.run(cmd + ["--dry-run"], cwd=str(root),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=180)
            rec["returncode"] = r.returncode
            rec["dry_run_output"] = (r.stdout or "")[-4000:]
            low = rec["dry_run_output"].lower()
            rec["accepted"] = (r.returncode == 0 and
                               "error" not in low and
                               "traceback" not in low)
            if not rec["accepted"]:
                rec["hint"] = "run_grid.py --dry-run rejected this plan; send pathA_probe.zip back"
        except Exception as e:
            rec["accepted"] = False
            rec["exception"] = repr(e)
        d["plan"].append(rec)

    # source-of-truth snippets for offline review
    for name in ("src/run_grid.py", "src/train_mvp.py", "配置.txt"):
        p = root / name
        try:
            d.setdefault("source_snippets", {})[name] = p.read_text(
                encoding="utf-8", errors="replace")[:6000]
        except Exception as e:
            d["errors"].append("%s: %r" % (name, e))

    # Pick the data root for the re-evaluation. The published CDT number is on
    # NEU-DET 10-shot, so prefer a ready NEU split; otherwise fall back to the
    # main DeepPCB split and flag that the comparison is cross-dataset.
    neu = [k for k, v in d["datasets"].items()
           if "neu" in k.lower() and v.get("shot10_ok")]
    other = [k for k, v in d["datasets"].items() if v.get("shot10_ok")]
    chosen = (neu or other or [None])[0]
    d["recommended"] = {
        "data_root": chosen,
        "same_dataset_as_paper": bool(neu),
        "reason": ("NEU-DET split found with complete shot10_s1..s%d" % DRAWS)
                  if neu else
                  ("No NEU-DET split with full shot10 draws; falling back to %s "
                   "(cross-dataset comparison, must be disclosed)" % chosen),
    }

    ok = sum(1 for x in d["plan"] if x["accepted"])
    d["ready"] = (ok == len(PLAN)) and d["runs"]["final_metrics"] > 0 and chosen is not None
    d["summary"] = {
        "plan_accepted": "%d/%d" % (ok, len(PLAN)),
        "final_metrics_on_server": d["runs"]["final_metrics"],
        "data_yamls_scanned": len(d["yamls"]),
        "training_yamls_skipped": skipped["training_args_yamls"] + skipped["under_runs_dir"],
        "non_string_values_skipped": len(skipped["non_string_values"]),
        "dataset_dirs": list(d["datasets"].keys()),
        "recommended_data_root": chosen,
        "same_dataset_as_paper": bool(neu),
        "ready_for_run": d["ready"],
    }
    Path(out_json).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=== PROBE SUMMARY ===")
    log(json.dumps(d["summary"], ensure_ascii=False, indent=2))
    for r in d["plan"]:
        log("  %-12s dry-run accepted=%s rc=%s" %
            (r["tag"], r.get("accepted"), r.get("returncode")))
    if not d["ready"]:
        log("")
        log("[STOP] Not ready. Do NOT use --run yet. Send pathA_probe.zip back.")
    return d


def run_all(root, args):
    pj = Path(args.probe_json)
    d = None
    if pj.is_file():
        try:
            d = json.loads(pj.read_text(encoding="utf-8"))
            if not d.get("ready"):
                log("[STOP] probe says ready=false; run --probe first and send it back")
                return 2
        except Exception as e:
            log("[WARN] probe json unreadable: %r" % e)
    else:
        log("[STOP] %s not found -- --probe has not completed successfully." % pj)
        log("       This lock is intentional: it prevents launching 20 runs blind.")
        log("       Fix: python /root/pathA.py --probe   (must print ready_for_run true)")
        return 2

    rec = (d.get("recommended") or {})
    data_root = args.data_root or rec.get("data_root") or ""
    log("DATA ROOT = %s" % (data_root or "(run_grid default)"))
    log("SAME DATASET AS PAPER = %s" % rec.get("same_dataset_as_paper"))
    if not rec.get("same_dataset_as_paper"):
        log("[NOTE] comparison is cross-dataset; disclose in the paper")
    extra = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.names:
        extra += ["--names", args.names]
    if data_root:
        extra += ["--data-root", str(data_root)]

    for item in PLAN:
        cmd = build_cmd(root, item, extra)
        log("")
        log("=== RUN %s ===" % item["tag"])
        log(" ".join(cmd))
        try:
            r = subprocess.run(cmd, cwd=str(root), check=False)
            log("returncode=%s" % r.returncode)
        except Exception as e:
            log("[ERROR] %r" % e)
    log("")
    log("=== DONE. Auto-packing results now ===")
    out_zip = ("/root/pathA_runs.zip" if args.out == "/root/pathA_probe.zip"
               else args.out)
    pack(root, args.probe_json, out_zip, "runs")
    log("=== FileZilla: pull the zip printed above and send it back ===")
    return 0


def pack(root, probe_json_path, out_zip, mode):
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    files = []

    pj = Path(probe_json_path)
    if pj.is_file():
        files.append((pj, "pathA_probe.json"))

    outdir = root / "输出"
    if outdir.is_dir():
        for p in sorted(outdir.glob(PROJECT_PREFIX + "*")):
            if not p.is_dir():
                continue
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() in SKIP_SUFFIX:
                    continue
                if any(part in SKIP_DIR for part in f.parts):
                    continue
                files.append((f, str(f.relative_to(root))))

    # always include a compact result index
    idx = []
    for f, arc in files:
        if f.name == "final_metrics.json":
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                idx.append({"file": arc,
                            "split": j.get("split"),
                            "mAP50": j.get("mAP50"),
                            "mAP50_95": j.get("mAP50_95"),
                            "config": j.get("config")})
            except Exception:
                idx.append({"file": arc, "parse_error": True})
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "root": str(root),
        "n_files": len(files) + 1,
        "n_final_metrics": len(idx),
        "result_index": idx,
        "note": "No .pt weights included. CDT re-evaluation of Ghali et al., IEEE SWC 2025.",
    }

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pathA_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for f, arc in files:
            try:
                z.write(f, arc)
            except OSError:
                pass
    log("=== PACK ===")
    log("zip: %s" % out_zip)
    log("size: %.1f KB" % (out_zip.stat().st_size / 1024.0))
    log("files: %d  (final_metrics: %d)" % (len(files) + 1, len(idx)))
    if not idx:
        log("[WARN] 0 results -- expected when --run has NOT completed yet.")
        log("       This is NOT a broken/empty-package bug: the meta-only zip is")
        log("       the correct output until the 20 runs finish. Do not send it back.")
        log("       Order: --probe  ->  --run  (that one auto-packs at the end)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--base", default="/root", help="where to start looking for ROOT")
    ap.add_argument("--probe-json", default="/root/pathA_probe.json")
    ap.add_argument("--out", default="/root/pathA_probe.zip")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--data-root", default="")
    a = ap.parse_args()

    root = None
    for b in ["/root/autodl-tmp/mvp", "/root/autodl-tmp", "/root", a.base]:
        root = find_root(Path(b))
        if root is not None:
            break
    if root is None:
        log("[FATAL] ROOT not found (no src/train_mvp.py). Try --base /root/autodl-tmp")
        return 3
    log("pathA.py %s  (expect %s; older v1 crashes on args.yaml)" % (VERSION, VERSION))
    log("ROOT = %s" % root)

    mode = "run" if a.run else ("pack" if a.pack else "probe")
    if a.probe or mode == "probe":
        out_zip = a.out if a.out.endswith(".zip") else "/root/pathA_probe.zip"
        d = probe(root, a.probe_json, a)
        return pack(root, a.probe_json, out_zip, "probe") if d["ready"] else pack(
            root, a.probe_json, out_zip, "probe_not_ready")

    if a.run:
        return run_all(root, a)

    if a.pack:
        out_zip = "/root/pathA_runs.zip" if a.out == "/root/pathA_probe.zip" else a.out
        return pack(root, a.probe_json, out_zip, "runs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
