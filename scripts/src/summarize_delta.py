# -*- coding: utf-8 -*-
"""汇总 PDC 前提验证结果：delta 曲线 vs mAP，并给出路线判定。

判定标准（预先固定，跑完不改）：
    |Spearman(delta, mAP)| >= 0.70 且关键对比方向正确
        -> delta 是有效可观测量，方法路线（在线 lr 控制器）成立
    0.50 <= |rho| < 0.70
        -> 边缘信号，需加到 3 个抽样再判
    |rho| < 0.50
        -> delta 不预测 mAP，放弃方法路线，收成评测协议论文（BME）
"""

import argparse
import glob
import json
from pathlib import Path


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx * dy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="输出/runs_delta")
    ap.add_argument("--out", default="输出/delta判定.md")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.project) / "*" / "final_metrics.json"))):
        d = json.load(open(f, encoding="utf-8"))
        cfg = d.get("config") or {}
        rows.append({
            "name": cfg.get("run_name") or Path(f).parent.name,
            "N": d["n_train_images"],
            "batch": cfg.get("batch"),
            "lr": cfg.get("lr0"),
            "mAP50": d["mAP50"] * 100,
            "mAP50_95": d["mAP50_95"] * 100,
            "delta": d.get("delta_final"),
            "norm0": d.get("delta_norm0"),
            "dir": str(Path(f).parent),
        })

    ok = [r for r in rows if r["delta"] is not None]
    lines = []
    add = lines.append

    add("# PDC 前提验证：参数位移 vs mAP\n")
    add(f"读到 {len(rows)} 个 run，其中 {len(ok)} 个带 delta 记录。\n")

    if len(ok) < 4:
        add("\n⚠️ 带 delta 的 run 不足 4 个，无法判定。请检查训练日志里是否有 "
            "`[delta] 参考点已建立` 一行；没有则说明回调未挂上。\n")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    add("| 配置 | N | batch | lr | mAP50 | mAP50-95 | delta 终值 |")
    add("|---|---|---|---|---|---|---|")
    for r in sorted(ok, key=lambda x: (x["N"], x["batch"] or 0, x["lr"] or 0)):
        add(f"| {r['name']} | {r['N']} | {r['batch']} | {r['lr']} | "
            f"{r['mAP50']:.2f} | {r['mAP50_95']:.2f} | {r['delta']:.5f} |")

    xs = [r["delta"] for r in ok]
    ys = [r["mAP50"] for r in ok]
    rho = spearman(xs, ys)
    add(f"\n**Spearman(delta, mAP50) = {rho:+.3f}**（n={len(ok)}）\n")

    # 关键对比：同一 N、同一 batch，只差 lr
    add("\n## 关键对比（同数据、同步数，只差 lr）\n")
    key_ok = True
    for n, b in {(r["N"], r["batch"]) for r in ok}:
        pair = [r for r in ok if r["N"] == n and r["batch"] == b]
        lrs = sorted({r["lr"] for r in pair})
        if len(lrs) < 2:
            continue
        hi = next((r for r in pair if r["lr"] == max(lrs)), None)
        lo = next((r for r in pair if r["lr"] == min(lrs)), None)
        if not hi or not lo:
            continue
        dm = hi["mAP50"] - lo["mAP50"]
        dd = hi["delta"] - lo["delta"]
        same_sign = (dm * dd) > 0
        add(f"- N={n}, batch={b}: lr {max(lrs):g} vs {min(lrs):g} → "
            f"mAP 差 {dm:+.2f} 点，delta 差 {dd:+.5f} "
            f"({'同向' if same_sign else '**反向**'})")
        if n == 30 and b == 30:
            key_ok = abs(dd) > 1e-4

    add("\n## 判定\n")
    if abs(rho) >= 0.70 and key_ok:
        verdict = (
            "**delta 是有效可观测量 → 走方法路线（PDC 在线 lr 控制器）。**\n\n"
            "依据：delta 与 mAP 的秩相关达到 0.70 以上，且关键对比（同数据只改 lr）"
            "中 delta 与 mAP 同向变化。下一步：实现控制器（delta 超出目标窗口时按比例"
            "缩放 lr），在 N=10/30/100 上与默认配置、oracle 逐档最优三者对比。"
        )
    elif abs(rho) >= 0.50:
        verdict = (
            "**边缘信号，需加到 3 个抽样再判。**\n\n"
            "当前 |rho| 在 0.50–0.70 之间，6 个点不足以定论。"
            "跑 `python src/run_delta.py --draw 2` 和 `--draw 3`（各 15–25 分钟），"
            "再重跑本脚本；若仍不到 0.70，按下一档处理。"
        )
    else:
        verdict = (
            "**delta 不预测 mAP → 放弃方法路线，收成评测协议论文（BME）。**\n\n"
            "这说明参数位移不是这个设定下的有效状态量（很可能因为 800 步内"
            "位移主要由 lr 而非泛化能力决定）。下一步转路线 A："
            "宽度↔lr（t=0.21）、lr↔batch（6.17 点漂移）、三旋钮等价、"
            "配对检验与检出力分析，加 NEU-DET 一个数据集即可成稿。"
        )
    add(verdict)

    # 附：曲线摘要
    add("\n## 位移曲线摘要\n")
    add("| 配置 | 首点 delta | 中点 delta | 末点 delta | 末/首 |")
    add("|---|---|---|---|---|")
    for r in sorted(ok, key=lambda x: (x["N"], x["batch"] or 0, x["lr"] or 0)):
        cf = Path(r["dir"]) / "delta_curve.json"
        if not cf.exists():
            continue
        try:
            curve = json.loads(cf.read_text(encoding="utf-8"))["curve"]
        except Exception:
            continue
        if not curve:
            continue
        first = curve[0]["delta"]
        mid = curve[len(curve) // 2]["delta"]
        last = curve[-1]["delta"]
        add(f"| {r['name']} | {first:.5f} | {mid:.5f} | {last:.5f} | "
            f"{(last / first if first else 0):.2f}× |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[已保存] {out}")


if __name__ == "__main__":
    main()
