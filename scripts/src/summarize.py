# -*- coding: utf-8 -*-
"""
汇总 runs_mvp 下所有 final_metrics.json -> 生成对比表 + 自动判定 (协议第 1 节标准)。
输出: mvp_results_summary.md (发回给 AI 用) + mvp_results_summary.csv
"""
import argparse
import csv
import json
from pathlib import Path

THRESH_GAIN_10SHOT = 2.0   # 低标注段增益门槛 (mAP50 点)
GAIN_RATIO = 2.0           # ΔG(10-shot) 须 >= GAIN_RATIO * ΔG(全量)
GROUP_DESC = {
    "g1": "G1 频域注意力",
    "g2": "G2 样本量自适应头",
    "g3": "G3 组合",
}


def load_runs(project="runs_mvp"):
    runs = {}
    for f in sorted(Path(project).glob("*/final_metrics.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            runs[f"{d['group']}_{d['split']}"] = d
        except Exception as e:
            print(f"[警告] 解析 {f} 失败: {e}")
    return runs


def agg(runs, group, prefix):
    """返回该组该档位所有 run 的 mAP50 列表"""
    vals = [d["mAP50"] for k, d in runs.items()
            if d["group"] == group and d["split"].startswith(prefix)]
    return vals


def fmt(vals):
    if not vals:
        return "缺"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{sum(vals) / len(vals):.4f} ±{(max(vals) - min(vals)):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs_mvp")
    ap.add_argument("--out-prefix", default="mvp_results_summary")
    args = ap.parse_args()
    runs = load_runs(args.project)
    if not runs:
        raise SystemExit(f"{args.project} 下没有 final_metrics.json, 请先运行 run_all.py")

    lines = ["# MVP 结果汇总", ""]
    lines.append("> 判定标准 (协议第 1 节): 10-shot 档 ΔG ≥ 2.0 mAP50, 且 ΔG(10-shot) ≥ 2×ΔG(全量)")
    lines.append("")

    # ---- 核心表 ----
    base10 = agg(runs, "g0", "shot10")
    base_full = agg(runs, "g0", "full")
    lines.append("## 核心对比表 (mAP50)")
    lines.append("")
    lines.append("| 组 | 10-shot (mean±range) | 全量 | ΔG(10-shot) | ΔG(全量) | 倍率 ΔG10/ΔG全 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    verdicts = {}
    for g in ("g1", "g2", "g3"):
        v10, vf = agg(runs, g, "shot10"), agg(runs, g, "full")
        if v10 and base10:
            d10 = sum(v10) / len(v10) - sum(base10) / len(base10)
        else:
            d10 = None
        if vf and base_full:
            dfull = sum(vf) / len(vf) - sum(base_full) / len(base_full)
        else:
            dfull = None
        if d10 is None or dfull is None:
            verdict, ratio = "数据不全", "-"
        else:
            ratio = "∞" if dfull <= 0 else f"{d10 / dfull:.2f}"
            # 判定阈值以 mAP 点为单位 (如 2.0 点), 而 d10/dfull 是小数 (如 0.0376) —— 先换算
            d10_pt, dfull_pt = d10 * 100, dfull * 100
            ok = (d10_pt >= THRESH_GAIN_10SHOT) and (dfull_pt <= 0 or d10_pt >= GAIN_RATIO * dfull_pt)
            verdict = "通过" if ok else "不通过"
        verdicts[g] = (d10, dfull)
        lines.append(
            f"| {GROUP_DESC.get(g, g)} | {fmt(v10)} | {fmt(vf)} | "
            f"{'%.4f' % d10 if d10 is not None else '缺'} | "
            f"{'%.4f' % dfull if dfull is not None else '缺'} | "
            f"{ratio} | {verdict} |"
        )
    lines.append("")

    # ---- mAP50-95 辅助表 ----
    def agg95(group, prefix):
        return [d["mAP50_95"] for k, d in runs.items()
                if d["group"] == group and d["split"].startswith(prefix)]
    lines.append("## 辅助表 (mAP50-95, 参考)")
    lines.append("")
    lines.append("| 组 | 10-shot | 全量 |")
    lines.append("|---|---|---|")
    for g in ("g0", "g1", "g2", "g3"):
        lines.append(f"| {GROUP_DESC.get(g, g) if g != 'g0' else 'G0 baseline'} | "
                     f"{fmt(agg95(g, 'shot10'))} | {fmt(agg95(g, 'full'))} |")
    lines.append("")

    # ---- 总判定 ----
    incomplete = any(
        verdicts.get(g, (None, None))[0] is None for g in ("g1", "g2", "g3")
    )
    passed = [g for g in ("g1", "g2") if verdicts.get(g, (None, None))[0] is not None
              and verdicts[g][0] * 100 >= THRESH_GAIN_10SHOT
              and (verdicts[g][1] is None or verdicts[g][1] <= 0
                   or verdicts[g][0] >= GAIN_RATIO * verdicts[g][1])]
    if incomplete:
        overall = "数据不全 —— 请跑完 16 个 run 后重新汇总 (python run_all.py)"
    elif len(passed) == 2:
        overall = "通过 —— 进入完整方案 (补 5/30 档 + 第二数据集)"
    elif len(passed) == 1:
        overall = f"部分通过 —— 仅 {'、'.join(passed)} 达标, 重设计未达标模块"
    else:
        overall = "不通过 —— Cheaper 故事不成立, 回到方案调整"
    lines.append(f"## 总判定: {overall}")
    lines.append("")
    n_seed = len(base10)
    if n_seed < 3:
        lines.append(f"> 注意: 当前 10-shot 只用了 **{n_seed} 个抽样种子**, 上表判定为方向性结论;"
                     f" 论文定稿前须补到 3 个种子（把 配置.txt 的「实验档位」改成 标准 重跑，已有结果自动跳过）。")
        lines.append("")

    # ---- 负交互检查 ----
    if all(verdicts.get(g, (None, None))[0] is not None for g in ("g1", "g2", "g3")):
        g3_10 = verdicts["g3"][0]
        if g3_10 * 100 < max(verdicts["g1"][0], verdicts["g2"][0]) * 100 - 1.0:
            lines.append("**警告: G3 组合出现负交互 (组合低于单模块最优 1 mAP50 以上), 需检查模块冲突。**")
            lines.append("")

    # ---- 参数量对照 (公平性检查) ----
    lines.append("## 参数量对照 (每次 run)")
    lines.append("")
    lines.append("| run | 参数量 | 备注 |")
    lines.append("|---|---|---|")
    for k, d in sorted(runs.items()):
        note = []
        if d.get("surgery", {}).get("head"):
            note.append(f"头缩放 s={d['surgery']['head']['scale']}, N={d['surgery']['head']['n_images']}")
        if d.get("surgery", {}).get("fca_layers"):
            note.append(f"FCA 层 {d['surgery']['fca_layers']}")
        lines.append(f"| {k} | {d.get('params', '?')} | {'; '.join(note)} |")
    lines.append("")

    # ---- 逐 run 明细 ----
    lines.append("## 逐 run 明细")
    lines.append("")
    lines.append("| run | N | mAP50 | mAP50-95 | epochs |")
    lines.append("|---|---|---|---|---|")
    for k, d in sorted(runs.items()):
        lines.append(f"| {k} | {d['n_train_images']} | {d['mAP50']} | {d['mAP50_95']} | "
                     f"{d.get('config', {}).get('epochs', '?')} |")
    lines.append("")

    lines.append("## 交回给 AI")
    lines.append("把本文件 (mvp_results_summary.md) 全文发回对话, 由 AI 按协议标准出最终结论与下一步建议。"
                 "如需更细诊断, 可附 runs_mvp/<run>/results.png (训练曲线) 与 *_finalval 的混淆矩阵。")

    out_md = Path(f"{args.out_prefix}.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    with open(f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "split", "n_train_images", "mAP50", "mAP50_95", "params"])
        for k, d in sorted(runs.items()):
            w.writerow([k, d["group"], d["split"], d["n_train_images"],
                        d["mAP50"], d["mAP50_95"], d.get("params", "")])

    print("\n".join(lines))
    print(f"\n已写出: {out_md.resolve()} 与 {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
