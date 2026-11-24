"""参数位移监测：记录训练过程中 ||theta_t - theta_0|| / ||theta_0||。

用途（PDC 前提验证）：
    判定"参数位移比 delta 能否预测最终 mAP"。
    若能（低 mAP 的配置 delta 明显更大/更小且单调），则可用 delta 做在线
    lr 控制器；若不能，则放弃方法路线，收成评测协议论文。

设计要点：
    - 只统计 requires_grad=True 的参数（全参数训练时即全部 11M）
    - theta_0 在 on_train_start 时抓取（此刻已完成头/注意力手术，是真正的起点）
    - theta_0 常驻 GPU（yolov8s 约 44 MB），避免每 K 步做主机-设备搬运
    - 任何异常都被吞掉并打印警告，绝不允许监测失败拖垮训练
"""

import json
from pathlib import Path


class DeltaMonitor:
    """记录 delta_t = ||theta_t - theta_0|| / ||theta_0|| 的曲线。"""

    def __init__(self, every=50, also_record_grad_norm=True):
        self.every = int(every)
        self.also_record_grad_norm = also_record_grad_norm
        self.theta0 = None
        self.norm0 = None
        self.step = 0
        self.records = []
        self._enabled = True

    # ---------- 回调 ----------

    def on_train_start(self, trainer):
        try:
            import torch
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            if not params:
                self._enabled = False
                print("[delta] 无可训练参数，监测关闭")
                return
            # clone + detach：防止优化器原地更新污染参考点
            self.theta0 = [p.detach().clone() for p in params]
            self.norm0 = torch.sqrt(
                sum((t.float() ** 2).sum() for t in self.theta0)
            ).item()
            lr0 = None
            if getattr(trainer, "optimizer", None) is not None:
                lr0 = trainer.optimizer.param_groups[0].get("lr")
            print(f"[delta] 参考点已建立: ||theta_0||={self.norm0:.4f}  lr0={lr0}  "
                  f"可训练参数={sum(p.numel() for p in params)}")
        except Exception as e:  # noqa: BLE001
            self._enabled = False
            print(f"[delta] 初始化失败，监测关闭: {e}")

    def on_train_batch_end(self, trainer):
        if not self._enabled:
            return
        self.step += 1
        if self.every <= 0 or self.step % self.every != 0:
            return
        try:
            import torch
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            with torch.no_grad():
                diff = torch.sqrt(
                    sum(((p - t0).float() ** 2).sum()
                        for p, t0 in zip(params, self.theta0))
                ).item()
            rec = {
                "step": self.step,
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "delta": round(diff / self.norm0, 6),
            }
            opt = getattr(trainer, "optimizer", None)
            if opt is not None:
                rec["lr"] = opt.param_groups[0].get("lr")
            if self.also_record_grad_norm:
                gnorm = 0.0
                for p in params:
                    if p.grad is not None:
                        gnorm += float((p.grad.detach().float() ** 2).sum().item())
                rec["grad_norm"] = round(gnorm ** 0.5, 6)
            self.records.append(rec)
        except Exception as e:  # noqa: BLE001
            self._enabled = False
            print(f"[delta] 记录失败，监测关闭: {e}")

    def on_train_end(self, trainer):
        if not self._enabled or not self.records:
            return
        try:
            save_dir = Path(getattr(trainer, "save_dir", "."))
            out = save_dir / "delta_curve.json"
            payload = {
                "norm_theta0": self.norm0,
                "every": self.every,
                "total_steps": self.step,
                "final_delta": self.records[-1]["delta"],
                "curve": self.records,
            }
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            print(f"[delta] 已保存 {out}  终值 delta={self.records[-1]['delta']:.5f}")
        except Exception as e:  # noqa: BLE001
            print(f"[delta] 保存失败: {e}")


def attach(YOLO_obj, every=50):
    """把监测器挂到 ultralytics YOLO 对象上。返回监测器实例。"""
    mon = DeltaMonitor(every=every)
    try:
        YOLO_obj.add_callback("on_train_start", mon.on_train_start)
        YOLO_obj.add_callback("on_train_batch_end", mon.on_train_batch_end)
        YOLO_obj.add_callback("on_train_end", mon.on_train_end)
    except Exception as e:  # noqa: BLE001
        print(f"[delta] 挂载回调失败（训练继续，不记录位移）: {e}")
    return mon
