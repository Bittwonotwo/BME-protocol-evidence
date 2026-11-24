# -*- coding: utf-8 -*-
"""
单次 MVP 训练。用法:
    python train_mvp.py --group g1 --split shot10_s1

group 定义:
    g0 = baseline (原版 YOLOv8s, 冻结配置)
    g1 = g0 + 频域注意力 (FcaNet 式 DCT 通道注意力, 插在 neck/head)
    g2 = g0 + 样本量自适应检测头 (隐藏宽度按 w(N)=sigmoid(a*log10(N)+b) 缩放)
    g3 = g1 + g2 组合

冻结原则 (协议第 6 节): 除组别模块与 N 之外, 所有超参在组间完全一致;
训练种子全程固定 seed=0, 随机性只来自 shot 子采样 (由 --split 区分)。
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _bootstrap  # noqa: E402  设好 KMP + 必要时切换 CUDA 解释器

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["g0", "g1", "g2", "g3"], required=True)
    ap.add_argument("--data-root", default="mvp_data")
    ap.add_argument("--split", required=True, help="full / shot10_s1 / shot10_s2 / shot10_s3")
    ap.add_argument("--model", default="", help="留空 = 使用包内 weights/yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = 自动 (shot 档 800, 全量档 100)")
    ap.add_argument("--train-val", action="store_true",
                    help="训练中每轮都验证。默认 shot 档关闭(快 6 倍, 结束时用密封测试集评一次, 取 last.pt);"
                         " 全量档始终开启")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 N 层; 0=不冻结")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs_mvp")
    ap.add_argument("--head-a", type=float, default=1.0, help="w(N) 的 a 参数 (冻结后勿改)")
    ap.add_argument("--head-b", type=float, default=-1.0, help="w(N) 的 b 参数 (冻结后勿改)")
    ap.add_argument("--head-scale", type=float, default=0.0,
                    help="网格实验用: 固定头宽度比例 0.25/0.5/0.75/1.0。"
                         "填了就不再用 w(N), 只按这个比例缩放检测头。0 = 用 w(N)")
    ap.add_argument("--run-name", default="",
                    help="自定义 run 目录名 (网格实验用, 默认 group_split)")
    ap.add_argument("--record-delta", type=int, default=0,
                    help="PDC 前提验证: 每 K 步记录 ||theta-theta0||/||theta0|| 到 "
                         "save_dir/delta_curve.json。0 = 关闭")
    args = ap.parse_args()

    # 单独运行本脚本时，若解释器用错了会自动切到 CUDA 环境（见 _bootstrap.py）
    _bootstrap.ensure_cuda(require=False)

    data_root = Path(args.data_root)
    data_yaml = data_root / f"{args.split}.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"找不到 {data_yaml}, 请先运行 prepare_data.py")
    train_dir = data_root / args.split / "images"
    n_images = len([p for p in train_dir.iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if n_images == 0:
        raise SystemExit(f"{train_dir} 下没有图片")

    # shot 档默认关闭"训练中验证": 10 张图的 epoch 只有 1 次梯度更新(约 0.05s),
    # 而验证要跑完整个测试集(约 1.5s) —— 97% 的时间花在画曲线上, 不是在训练。
    # 关闭后单 run 从约 5 分钟降到约 50 秒, 省下的预算用来把梯度更新次数从 200 提到 800。
    is_shot = args.split.startswith("shot")
    do_train_val = args.train_val or not is_shot
    epochs = args.epochs or ((800 if not do_train_val else 200) if is_shot else 100)
    batch = max(1, min(args.batch, n_images))
    model_path = args.model or str(HERE.parent / "weights" / "yolov8s.pt")
    if not Path(model_path).exists():
        raise SystemExit(f"找不到预训练权重 {model_path}, 请确认 weights/yolov8s.pt 存在")
    run_name = args.run_name or f"{args.group}_{args.split}"
    out_dir = Path(args.project) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "final_metrics.json"
    if marker.exists():
        # 续跑时不能只看"文件在" —— 早期冒烟测试留下的 1-epoch 结果必须重跑;
        # 网格实验里同一 run 名换了缩放比例也必须重跑。
        try:
            old = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        old_ep = old.get("config", {}).get("epochs")
        old_sc = old.get("config", {}).get("head_scale_requested", 0.0)
        if old_ep == epochs and float(old_sc or 0) == float(args.head_scale):
            print(f"[跳过] {run_name} 已有 final_metrics.json (epochs={epochs})")
            return
        print(f"[重跑] {run_name} 旧结果不符 (epochs {old_ep}→{epochs}, "
              f"head_scale {old_sc}→{args.head_scale}), 删除旧结果")
        import shutil as _sh
        _sh.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 关键: 拦截 ultralytics 的 get_model, 在模型构建+预训练权重载入之后
    # 立刻施加 G1/G2 改造, 这样优化器/EMA 看到的就是改造后的模型 ----------
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from mvp_modules import apply_fca, apply_head_scaling

    orig_get_model = DetectionTrainer.get_model
    surgery_log = {}

    def patched_get_model(self, *a, **k):
        m = orig_get_model(self, *a, **k)
        if args.group in ("g1", "g3"):
            surgery_log["fca_layers"] = apply_fca(m)
        if args.head_scale > 0:
            # 网格实验: 固定比例缩放检测头 (不用 w(N))
            surgery_log["head"] = apply_head_scaling(m, scale=args.head_scale)
        elif args.group in ("g2", "g3"):
            surgery_log["head"] = apply_head_scaling(
                m, n_images, a=args.head_a, b=args.head_b
            )
        surgery_log["params"] = sum(p.numel() for p in m.parameters())
        return m

    DetectionTrainer.get_model = patched_get_model

    print(f"[run] group={args.group} split={args.split} N={n_images} epochs={epochs} "
          f"训练中验证={'开' if do_train_val else '关'}")
    device = args.device
    if device not in ("cpu", "mps") and not torch.cuda.is_available():
        print("[警告] 未检测到可用 CUDA, 自动改用 CPU 训练(非常慢)。")
        print("       如果这是正式实验, 请先安装 CUDA 版 torch (运行 python run.py check 可看提示)。")
        device = "cpu"

    # 显存预检: 本机 RTX 3060 Laptop 6GB 实测 —— 训练峰值 reserved 满足
    #   reserved ≈ 0.9~1.2GB (固定开销/上下文) + 0.17~0.20GB × batch
    #   (batch=8 → 2.26GB, batch=16 → 3.54GB, 两条实测点)
    # 开跑前按"当前真正空闲的显存"倒推能承受的 batch (留余量), 避免撞上 OOM 把 CUDA 上下文搞坏。
    if device not in ("cpu", "mps") and torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        afford = max(1, int((free_gb * 0.85 - 1.2) / 0.20))
        if batch > afford:
            print(f"[显存预检] 当前空闲 {free_gb:.1f} GB, batch={batch} 预计需 "
                  f"{1.2 + 0.20 * batch:.1f} GB → 自动降到 batch={afford}")
            batch = afford

    print(f"[run] batch={batch} device={device} epochs={epochs}")

    def _train_once(b):
        m = YOLO(model_path)
        if args.record_delta > 0:
            # PDC 前提验证: 记录参数位移曲线, 失败不影响训练
            import delta_monitor
            delta_monitor.attach(m, every=args.record_delta)
        m.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=b,
            device=device,
            workers=args.workers,
            seed=0,               # 训练种子全程固定 —— 公平性红线
            deterministic=True,
            optimizer="SGD",
            lr0=args.lr0,
            freeze=args.freeze if args.freeze > 0 else None,
            patience=1000,        # 禁用早停, 防止各组停止点不同造成不公平
            val=do_train_val,     # shot 档关闭训练中验证 (见上方注释)
            project=args.project,
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        return m

    # 显存保护: 万一 batch 超出显卡容量, 自动减半重试, 不会让整个实验因为一次 OOM 中断。
    batch_used = batch
    tried = []
    while True:
        try:
            model = _train_once(batch_used)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg or batch_used <= 1:
                raise
            nxt = max(1, batch_used // 2)
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 0
            print(f"\n[显存不足] batch={batch_used} 装不下(当前可用 {free_gb:.1f} GB), "
                  f"自动降到 batch={nxt} 重跑本组。建议把 配置.txt 的 batch 改成 {nxt}。")
            tried.append(batch_used)
            torch.cuda.empty_cache()
            batch_used = nxt

    peak_gb = None
    # ultralytics 内部也可能自己把 batch 减半(它捕获 OOM 后重试), 这里以 trainer 实际用的为准,
    # 否则记录下来的 batch 会和真实训练不符, 影响实验可复现性。
    actual_batch = getattr(getattr(model, "trainer", None), "batch_size", batch_used)
    if actual_batch != batch_used:
        print(f"[提示] ultralytics 内部把 batch 从 {batch_used} 自动降到了 {actual_batch}, "
              f"建议把 配置.txt 的 batch 改成 {actual_batch}。")
        batch_used = actual_batch
    if device not in ("cpu", "mps") and torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        print(f"[显存] 本次训练峰值 reserved={peak_gb} GB (batch={batch_used})")

    # ---------- 在密封测试集上做最终评估 ----------
    # 注意: 不同版本 ultralytics 的 project 参数解释可能不同 (可能嵌套在全局 runs 目录下),
    # 因此以 trainer 实际的 save_dir 为准, 不自行拼接路径。
    # 训练中验证开启时用 best.pt (按验证集挑模型); 关闭时用 last.pt (固定步数的终点, 无挑选偏差)。
    save_dir = Path(model.trainer.save_dir)
    ckpt_order = ["best.pt", "last.pt"] if do_train_val else ["last.pt", "best.pt"]
    best = next((save_dir / "weights" / c for c in ckpt_order
                 if (save_dir / "weights" / c).exists()), None)
    if best is None:
        raise SystemExit(f"训练完成但未找到权重文件, 请检查 {save_dir / 'weights'}")
    print(f"[eval] 使用 {best.name} 在密封测试集上评估")
    m2 = YOLO(str(best))
    v = m2.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        device=device,
        plots=True,
        project=args.project,
        name=run_name + "_finalval",
        exist_ok=True,
    )

    names = m2.model.names if hasattr(m2.model, "names") else {}
    per_class = {}
    try:
        for ci, ap50 in zip(list(v.box.ap_class_index), list(v.box.ap50)):
            per_class[str(names.get(int(ci), ci))] = round(float(ap50), 4)
    except Exception:
        pass

    metrics = {
        "group": args.group,
        "split": args.split,
        "n_train_images": n_images,
        "mAP50": round(float(v.box.map50), 4),
        "mAP50_95": round(float(v.box.map), 4),
        "per_class_ap50": per_class,
        "params": sum(p.numel() for p in m2.model.parameters()),
        "surgery": surgery_log,
        "peak_vram_gb": peak_gb,
        "config": {
            "epochs": epochs, "batch": batch_used, "batch_requested": args.batch,
            "oom_downgraded": tried, "imgsz": args.imgsz,
            "lr0": args.lr0, "freeze": args.freeze, "model": Path(model_path).name, "seed": 0,
            "head_a": args.head_a, "head_b": args.head_b,
            "head_scale_requested": args.head_scale,
            "run_name": run_name,
            "train_val": do_train_val, "ckpt": best.name,
        },
    }
    # PDC 前提验证: 把位移曲线终值写进 metrics, 汇总时可直接与 mAP 对齐
    if args.record_delta > 0:
        dc = save_dir / "delta_curve.json"
        if dc.exists():
            try:
                payload = json.loads(dc.read_text(encoding="utf-8"))
                metrics["delta_final"] = payload.get("final_delta")
                metrics["delta_norm0"] = payload.get("norm_theta0")
                metrics["delta_every"] = payload.get("every")
            except Exception as e:  # noqa: BLE001
                print(f"[delta] metrics 读取曲线失败: {e}")
        else:
            print("[delta] 未找到 delta_curve.json（监测可能初始化失败）")

    marker.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] {run_name}: mAP50={metrics['mAP50']} mAP50-95={metrics['mAP50_95']} -> {marker}")


if __name__ == "__main__":
    main()
