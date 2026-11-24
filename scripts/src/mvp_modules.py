# -*- coding: utf-8 -*-
"""
MVP 核心模块
G1: FcaAttention / FcaWrap —— FcaNet 式多频谱(DCT)通道注意力
    参考: Qin et al., "FcaNet: Frequency Channel Attention Networks", ICCV 2021
G2: width_scale / apply_head_scaling —— 样本量自适应检测头
    机制: 检测头隐藏宽度按 w(N) = sigmoid(a*log10(N)+b) 缩放, N = 训练集图像数。
    单一函数覆盖所有标注档位, 禁止逐档手调 —— 这是区分"机制"与"调参"的金标准。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def width_scale(n_images, a=1.0, b=-1.0):
    """G2 核心函数: 头容量缩放系数。N=10 -> 0.5, N=1000 -> 0.88, N=10000 -> 0.95"""
    return 1.0 / (1.0 + math.exp(-(a * math.log10(max(int(n_images), 1)) + b)))


class FcaAttention(nn.Module):
    """FcaNet 式多频谱通道注意力: 通道分组 -> 空间池化 -> 每组一个 DCT 频率分量 -> FC -> sigmoid 门控"""

    def __init__(self, channels, groups=4, pool_size=8, reduction=16):
        super().__init__()
        c = int(channels)
        assert c > 0
        self.pool_size = pool_size
        # 选一个能整除通道数的组数
        self.g = next(g for g in (groups, 2, 1) if c % g == 0)
        # 前 g 个 zigzag 频率分量: DC, 水平1, 垂直1, 对角1
        freqs = [(0, 0), (0, 1), (1, 0), (1, 1)][: self.g]
        n = pool_size
        basis = torch.zeros(self.g, n, n)
        for k, (p, q) in enumerate(freqs):
            ap = math.sqrt(1.0 / n) if p == 0 else math.sqrt(2.0 / n)
            aq = math.sqrt(1.0 / n) if q == 0 else math.sqrt(2.0 / n)
            for u in range(n):
                for v in range(n):
                    basis[k, u, v] = (
                        ap
                        * aq
                        * math.cos(math.pi * (2 * u + 1) * p / (2 * n))
                        * math.cos(math.pi * (2 * v + 1) * q / (2 * n))
                    )
        self.register_buffer("basis", basis)
        hidden = max(c // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, c)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))  # b,c,n,n
        xs = pooled.view(b, self.g, c // self.g, self.pool_size, self.pool_size)
        # 每组用各自的 DCT 基提取一个频率分量系数
        y = (xs * self.basis.view(1, self.g, 1, self.pool_size, self.pool_size)).sum(
            dim=(-2, -1)
        )  # b, g, c/g
        w = torch.sigmoid(self.fc(y.reshape(b, c))).view(b, c, 1, 1)
        return x * w


class FcaWrap(nn.Module):
    """包装 ultralytics 的 SPPF/C2f 层: 原层输出后接 FcaAttention(通道数不变, 接口安全)"""

    def __init__(self, inner):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv as UConv

        out_ch = inner.cv2.conv.out_channels  # SPPF 与 C2f 的输出卷积都是 .cv2
        self.inner = inner
        self.fca = FcaAttention(out_ch)
        # 复制 DetectionModel 前向循环需要的元数据
        for attr in ("f", "i", "type", "np"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def forward(self, x):
        return self.fca(self.inner(x))


def apply_fca(det_model):
    """G1: 包装 neck/head 区域的 SPPF 与 C2f 层(backbone 不动)。返回被包装的层索引。"""
    wrapped = []
    for i, m in enumerate(det_model.model):
        name = type(m).__name__
        if name == "SPPF" or (name == "C2f" and i >= 10):
            det_model.model[i] = FcaWrap(m)
            wrapped.append(i)
    if not wrapped:
        raise RuntimeError("未找到可包装的 SPPF/C2f 层, 请检查模型结构 (期望 YOLOv8 系)")
    return wrapped


def _slice_uconv(conv, keep_in=None, keep_out=None):
    """复制一个 ultralytics Conv 并按通道切片(保留预训练权重)"""
    from ultralytics.nn.modules.conv import Conv as UConv

    c1 = conv.conv.in_channels if keep_in is None else keep_in
    c2 = conv.conv.out_channels if keep_out is None else keep_out
    k = conv.conv.kernel_size[0]
    s = conv.conv.stride[0]
    new = UConv(c1, c2, k, s)
    with torch.no_grad():
        new.conv.weight.copy_(conv.conv.weight[:c2, :c1])
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(new.bn, attr).copy_(getattr(conv.bn, attr)[:c2])
    return new


def _slice_plain_conv(conv, keep_in=None):
    """复制一个普通 nn.Conv2d 并裁剪输入通道"""
    c1 = conv.in_channels if keep_in is None else keep_in
    c2 = conv.out_channels
    new = nn.Conv2d(
        c1, c2, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
        conv.groups, bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[:c2, :c1])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def apply_head_scaling(det_model, n_images=None, a=1.0, b=-1.0, min_width=16, scale=None):
    """对 Detect 头 (cv2 框回归分支 / cv3 分类分支) 的隐藏宽度做缩放。

    两种用法:
      * 自适应 (原 G2): 传 n_images, 缩放系数由 w(N) 算出 —— 机制主张。
      * 固定比例 (网格实验用): 传 scale=0.25/0.5/0.75/1.0, 直接指定宽度比例 ——
        用来扫 s*(N) 曲面, 回答"窄头到底有没有用、在哪个 N 有用"。

    只缩放隐藏层, 输出维度 (reg_max*16 / nc) 不变, 不影响 DFL 解码。返回缩放信息。
    """
    if scale is not None:
        s = float(scale)
    else:
        s = width_scale(n_images, a, b)
    head = det_model.model[-1]
    if type(head).__name__ != "Detect":
        raise RuntimeError(f"期望 Detect 头, 实际为 {type(head).__name__}")
    for branch_name in ("cv2", "cv3"):
        branch = getattr(head, branch_name)
        for lvl in range(len(branch)):
            seq = branch[lvl]
            mods = list(seq.children())
            ok = (
                len(mods) == 3
                and type(mods[0]).__name__ == "Conv"
                and type(mods[1]).__name__ == "Conv"
                and isinstance(mods[2], nn.Conv2d)
                and mods[2].kernel_size == (1, 1)
            )
            if not ok:
                raise RuntimeError(
                    "检测头结构与预期不符 (期望 [Conv3x3, Conv3x3, Conv2d-1x1])。"
                    "请将 ultralytics 固定到 8.3.x 系列版本后重试。"
                )
            h = mods[0].conv.out_channels
            h_new = max(min_width, round(h * s))
            if h_new == h:
                continue
            branch[lvl] = nn.Sequential(
                _slice_uconv(mods[0], keep_out=h_new),
                _slice_uconv(mods[1], keep_in=h_new, keep_out=h_new),
                _slice_plain_conv(mods[2], keep_in=h_new),
            )
    return {
        "scale": round(s, 4),
        "n_images": int(n_images) if n_images is not None else None,
        "a": a,
        "b": b,
        "fixed_scale": scale is not None,
        "params_after": sum(p.numel() for p in det_model.parameters()),
    }
