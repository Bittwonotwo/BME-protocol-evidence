# -*- coding: utf-8 -*-
"""网格结果汇总: 读 输出/runs_grid/, 生成 mAP = f(s, N) 表格 + s*(N) 判定。

用法:
    python src/summarize_grid.py
    python src/summarize_grid.py --project 输出/runs_grid --out 输出/网格结果.md

判定标准(写死, 防止事后找理由):
    * 平坦   : 所有 N 档里, 最优宽度相对 s=1.0 的增益 < 1.5 点
               -> 检测头容量对标注量不敏感, "容量-数据匹配"假设不成立, 应放弃该 idea
    * 有结构 : 至少一个 N 档增益 >= 2.0 点, 且该档里 >=80% 的抽样方向一致
               -> 曲面有拐点, 值得铺全网格并拟合 w(N)
    * 弱信号 : 其余情况 -> 加抽样数(或换 N 档)再判, 不要直接写论文
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(ROOT / "输出" / "runs_grid"))
    ap.add_argument("--out", default=str(ROOT / "输出" / "网格结果.md"))
    args = ap.parse_args()

    proj = Path(args.project)
    if not proj.exists():
        raise SystemExit(f"[失败] 找不到 {proj}, 请先运行 src/run_grid.py")

    runs = []
    for m in sorted(proj.glob("*/final_metrics.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        head = (d.get("surgery") or {}).get("head") or {}
        s = head.get("scale")
        if s is None:
            s = d.get("config", {}).get("head_scale_requested")
        if not s:
            continue
        n = int(d.get("n_train_images") or 0)
        if n <= 0:
            continue
        runs.append({
            "n": n, "scale": round(float(s), 4), "split": d.get("split", ""),
            "mAP50": float(d["mAP50"]), "mAP": float(d["mAP50_95"]),
            "params": d.get("params"), "per_class": d.get("per_class_ap50") or {},
            "name": m.parent.name,
        })
    if not runs:
        raise SystemExit(f"[失败] {proj} 下没有可用的网格结果 (缺少 head.scale 字段)")

    Ns = sorted({r["n"] for r in runs})
    scales = sorted({r["scale"] for r in runs})
    by = {(r["n"], r["scale"], r["split"]): r for r in runs}

    def vals(n, s):
        return [r["mAP50"] * 100 for r in runs if r["n"] == n and r["scale"] == s]

    lines = []
    P = lines.append
    P("# 网格实验结果：检测头宽度 s × 标注量 N")
    P("")
    P(f"数据来源：`{proj}`（{len(runs)} 个 run，指标为密封测试集 mAP50，单位 %）")
    P("")

    # ---------- 表1: 均值 ----------
    P("## 一、mAP50 均值（%）")
    P("")
    P("| N \\ s | " + " | ".join(f"{s:.2f}" for s in scales) + " | 最优 s | 增益(点) |")
    P("|---" * (len(scales) + 3) + "|")
    summary = {}
    for n in Ns:
        row, best_s, best_v, base_v = [], None, -1, None
        for s in scales:
            v = vals(n, s)
            if not v:
                row.append("—")
                continue
            mu = st.mean(v)
            row.append(f"{mu:.2f}" + (f" (n={len(v)})" if len(v) > 1 else ""))
            if s == 1.0:
                base_v = mu
            if mu > best_v:
                best_v, best_s = mu, s
        gain = None if base_v is None else best_v - base_v
        summary[n] = {"best_s": best_s, "best_v": best_v, "base_v": base_v, "gain": gain}
        P(f"| {n} | " + " | ".join(row) + f" | {best_s:.2f} | {fmt(gain)} |")
    P("")

    # ---------- 表2: 配对差与方向一致性 ----------
    P("## 二、最优宽度 vs s=1.0 的配对差（同抽样相减）")
    P("")
    P("| N | 最优 s | 抽样数 | 逐抽样差值(点) | 均值 | 标准差 | 方向一致 |")
    P("|---|---|---|---|---|---|---|")
    verdict_rows = []
    for n in Ns:
        s_star = summary[n]["best_s"]
        if s_star is None or abs(s_star - 1.0) < 1e-9:
            P(f"| {n} | 1.00 | — | — | 0.00 | — | 基线即最优 |")
            verdict_rows.append((n, 0.0, 0, 0))
            continue
        splits = sorted({r["split"] for r in runs if r["n"] == n})
        diffs = []
        for sp in splits:
            a, b = by.get((n, s_star, sp)), by.get((n, 1.0, sp))
            if a and b:
                diffs.append((a["mAP50"] - b["mAP50"]) * 100)
        if not diffs:
            P(f"| {n} | {s_star:.2f} | 0 | — | — | — | — |")
            verdict_rows.append((n, None, 0, 0))
            continue
        mu = st.mean(diffs)
        sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
        win = sum(1 for d in diffs if d > 0)
        P(f"| {n} | {s_star:.2f} | {len(diffs)} | "
          f"{', '.join(f'{d:+.2f}' for d in diffs)} | {mu:+.2f} | {fmt(sd)} | {win}/{len(diffs)} |")
        verdict_rows.append((n, mu, win, len(diffs)))
    P("")

    # ---------- 表3: 逐类 ----------
    if all(r["per_class"] for r in runs):
        P("## 三、各类 AP50 在最优宽度下的变化（均值，点）")
        P("")
        cls = sorted(runs[0]["per_class"].keys())
        P("| 类别 | " + " | ".join(f"N={n}" for n in Ns) + " |")
        P("|---" * (len(Ns) + 1) + "|")
        for c in cls:
            cells = []
            for n in Ns:
                s_star = summary[n]["best_s"]
                if s_star is None:
                    cells.append("—")
                    continue
                a = [r["per_class"].get(c) for r in runs if r["n"] == n and r["scale"] == s_star and c in r["per_class"]]
                b = [r["per_class"].get(c) for r in runs if r["n"] == n and r["scale"] == 1.0 and c in r["per_class"]]
                a = [x for x in a if x is not None]
                b = [x for x in b if x is not None]
                cells.append(f"{(st.mean(a) - st.mean(b)) * 100:+.2f}" if a and b else "—")
            P(f"| {c} | " + " | ".join(cells) + " |")
        P("")

    # ---------- 判定 ----------
    gains = [g for (_, g, w, t) in verdict_rows if g is not None]
    structured = [(n, g, w, t) for (n, g, w, t) in verdict_rows
                  if g is not None and g >= 2.0 and t > 0 and w / t >= 0.8]
    if gains and max(gains) < 1.5:
        verdict = "平坦"
        emoji = "【判定：平坦 → 假设不成立】"
        action = ("所有 N 档里最优宽度相对 s=1.0 的增益都 < 1.5 点，说明检测头容量对标注量"
                  "不敏感。这个 idea 的机制假设被否定，应立即停止投入，换创新点。")
    elif structured:
        verdict = "有结构"
        emoji = "【判定：有结构 → 值得铺全网格】"
        action = ("至少一个 N 档增益 ≥ 2.0 点且方向一致（≥80% 抽样），曲面存在拐点。"
                  "下一步跑 `python src/run_grid.py --stage full` 补齐 N=100/300/全量，"
                  "再拟合 s*(N) 得到闭式 w(N)。")
    else:
        verdict = "弱信号"
        emoji = "【判定：弱信号 → 加抽样再判】"
        action = ("有正向趋势但未达判定线。把每个 N 的抽样数加到 8–10 个再跑一次"
                  "（成本约翻倍），仍不达标就按\"平坦\"处理。")

    P("## 四、判定")
    P("")
    P(f"**{emoji}**")
    P("")
    P(action)
    P("")
    P("判定线（预先写死）：平坦 = 全部增益 < 1.5 点；有结构 = 某档 ≥ 2.0 点且 ≥80% 抽样同向。")
    P("")

    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[完成] 已写入 {out}")


if __name__ == "__main__":
    main()
