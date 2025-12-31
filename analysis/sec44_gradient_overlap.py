"""
§4.4 Gradient Overlap Analysis
================================
对所有架构（FNO / CNO / DeepONet），计算：

    r_low  = ||U^T g_low ||² / ||g_low ||²
    r_high = ||U^T g_high||² / ||g_high||²

其中：
    U       = Fisher 子空间基向量（来自 null_space_projections.pt）
    g_low   = 低频 loss 对参数的梯度（pretrained 模型在 real val 数据上）
    g_high  = 高频 loss 对参数的梯度

如果 r_low > r_high：说明 Fisher 子空间主要"对齐"物理低频梯度
—— 这是架构无关的最直接证明

输出：
    analysis/figures/fig_gradient_overlap.pdf / .png
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────
ROOT = Path("./RealPDEBench")
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 每个架构的配置 ──────────────────────────────────
ARCHS = {
    "FNO": {
        "proj_pt": ROOT / "results/001-cylinder/fno/fno_cylinder_nsft/2026-03-10_08-42-11/null_space_projections.pt",
        "ckpt":    ROOT / "results/001-cylinder/fno/fno_cylinder_pretrained/2026-03-09_17-33-29/model_3760.pth",
        "config":  ROOT / "realpdebench/configs/cylinder/fno_nullspace.yaml",
    },
    "CNO": {
        "proj_pt": ROOT / "results/001-cylinder/cno/cno_cylinder_nsft/2026-03-14_06-37-42/null_space_projections.pt",
        "ckpt":    ROOT / "results/001-cylinder/cno/cno_cylinder_pretrained/2026-03-11_09-49-04/model_4300.pth",
        "config":  ROOT / "realpdebench/configs/cylinder/cno.yaml",
    },
    "DeepONet": {
        "proj_pt": ROOT / "results/001-cylinder/deeponet/deeponet_cylinder_nsft/2026-03-13_00-38-45/null_space_projections.pt",
        "ckpt":    ROOT / "results/001-cylinder/deeponet/deeponet_cylinder_pretrained/2026-03-12_11-16-03/model_5000.pth",
        "config":  ROOT / "realpdebench/configs/cylinder/deeponet.yaml",
    },
}

N_SAMPLES = 16   # CNO 显存需求大，用更少样本
DEVICE    = "cuda:2" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────
# 辅助：频率分离 low / high
# ─────────────────────────────────────────────────────
def split_freq(pred: torch.Tensor, low_cutoff: float = 0.2):
    """
    pred: [B, T, H, W, C] 或 [B, H, W, C]
    在空间维度上做 FFT，低频截断后 IFFT 得到 pred_low，
    pred_high = pred - pred_low
    low_cutoff: 0..1, 保留的频率比例
    """
    # 取最后三维中的空间两维
    shape = pred.shape
    if pred.ndim == 5:
        # [B, T, H, W, C] -> process H, W
        B, T, H, W, C = shape
        x = pred.permute(0, 1, 4, 2, 3).reshape(B * T * C, H, W)
    else:
        B, H, W, C = shape
        x = pred.permute(0, 3, 1, 2).reshape(B * C, H, W)

    X = torch.fft.fft2(x)

    # 构建低频 mask
    ny, nx = H, W
    mask = torch.zeros(ny, nx, dtype=torch.bool, device=x.device)
    cy, cx = int(ny * low_cutoff), int(nx * low_cutoff)
    mask[:cy, :cx] = True
    mask[-cy:, :cx] = True
    mask[:cy, -cx:] = True
    mask[-cy:, -cx:] = True

    X_low  = X * mask
    X_high = X * (~mask)

    x_low  = torch.fft.ifft2(X_low).real
    x_high = torch.fft.ifft2(X_high).real

    if pred.ndim == 5:
        x_low  = x_low.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)
        x_high = x_high.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)
    else:
        x_low  = x_low.reshape(B, C, H, W).permute(0, 2, 3, 1)
        x_high = x_high.reshape(B, C, H, W).permute(0, 2, 3, 1)

    return x_low.contiguous(), x_high.contiguous()


# ─────────────────────────────────────────────────────
# 计算单个架构的梯度重叠率
# ─────────────────────────────────────────────────────
def compute_overlap(arch_name: str, cfg: dict, val_loader) -> dict:
    print(f"\n  [{arch_name}] 加载模型 ...")
    # 加载投影矩阵
    proj_data = torch.load(cfg["proj_pt"], map_location="cpu", weights_only=False)
    projections = proj_data["projections"]  # {layer_name: {"U": tensor, "dim": int}}

    # 加载 config
    from realpdebench.model.load_model import load_model
    import yaml

    with open(cfg["config"]) as f:
        config = yaml.safe_load(f)

    # 覆盖 dataset_root 为当前机器路径
    config["dataset_root"] = "./data/realpdebench"

    # 用 val_loader 的 dataset 来确定 input/output shape
    model = load_model(val_loader.dataset, device=DEVICE, **config)
    model.load_checkpoint(str(cfg["ckpt"]), DEVICE)
    model.eval()

    # 确定要分析的 layers（存在于 projections 中的）
    protected = {n: p for n, p in projections.items() if "U" in p}

    # 累积梯度
    sum_g_low  = {}
    sum_g_high = {}
    for name in protected:
        sum_g_low[name]  = None
        sum_g_high[name] = None

    n_done = 0
    for batch in val_loader:
        if n_done >= N_SAMPLES:
            break
        inp, tgt = batch[0].to(DEVICE), batch[1].to(DEVICE)

        # 低频 loss
        model.zero_grad()
        pred = model(inp)
        tgt_low, _ = split_freq(tgt)
        pred_low, _ = split_freq(pred.detach().clone())
        # 重新 forward，保留 grad，只对低频部分求 loss
        pred2 = model(inp)
        pred2_low, pred2_high = split_freq(pred2)
        loss_low  = nn.functional.mse_loss(pred2_low,  tgt_low)
        loss_low.backward()
        for name, p in model.named_parameters():
            if name in protected and p.grad is not None:
                g = p.grad.detach().reshape(-1)
                if g.is_complex():
                    g = torch.cat([g.real, g.imag])
                g = g.float().cpu()
                sum_g_low[name] = g if sum_g_low[name] is None else sum_g_low[name] + g

        # 高频 loss
        model.zero_grad()
        pred3 = model(inp)
        _, pred3_high = split_freq(pred3)
        _, tgt_high   = split_freq(tgt)
        loss_high = nn.functional.mse_loss(pred3_high, tgt_high)
        loss_high.backward()
        for name, p in model.named_parameters():
            if name in protected and p.grad is not None:
                g = p.grad.detach().reshape(-1)
                if g.is_complex():
                    g = torch.cat([g.real, g.imag])
                g = g.float().cpu()
                sum_g_high[name] = g if sum_g_high[name] is None else sum_g_high[name] + g

        n_done += 1
        if n_done % 8 == 0:
            print(f"    {n_done}/{N_SAMPLES} samples done")

    print(f"  [{arch_name}] 计算重叠率 ...")
    r_low_list, r_high_list = [], []

    for name, pdata in protected.items():
        if sum_g_low.get(name) is None:
            continue
        U = pdata["U"].float()   # [d, k]
        g_low  = sum_g_low[name]
        g_high = sum_g_high[name]

        if g_low.shape[0] != U.shape[0]:
            continue

        # 投影比例 r = ||U^T g||² / ||g||²
        def overlap(g, U):
            proj = U.T @ g       # [k]
            return (proj ** 2).sum() / ((g ** 2).sum() + 1e-12)

        r_low_list.append(overlap(g_low, U).item())
        r_high_list.append(overlap(g_high, U).item())

    # 释放 GPU 显存
    del model
    torch.cuda.empty_cache()

    return {
        "r_low":  float(np.mean(r_low_list)),
        "r_high": float(np.mean(r_high_list)),
        "n_layers": len(r_low_list),
    }


# ─────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────
def main():
    # 加载 val 数据
    print("加载 Cylinder real val 数据 ...")
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from torch.utils.data import DataLoader

    DATA_ROOT = "./data/realpdebench"
    val_ds = CylinderHFDataset(
        dataset_name="cylinder",
        dataset_root=DATA_ROOT,
        dataset_type="real",
        mode="val",
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"  Val size: {len(val_ds)} samples")

    results = {}
    for arch_name, cfg in ARCHS.items():
        if not Path(cfg["proj_pt"]).exists():
            print(f"  [{arch_name}] SKIP — proj_pt not found")
            continue
        if not Path(cfg["ckpt"]).exists():
            print(f"  [{arch_name}] SKIP — ckpt not found")
            continue
        try:
            results[arch_name] = compute_overlap(arch_name, cfg, val_loader)
        except Exception as e:
            print(f"  [{arch_name}] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            # 及时释放 GPU 显存，避免 OOM
            torch.cuda.empty_cache()

    if not results:
        print("没有任何架构成功计算，退出。")
        return

    # ── 画图 ──────────────────────────────────────────
    archs   = list(results.keys())
    r_lows  = [results[a]["r_low"]  for a in archs]
    r_highs = [results[a]["r_high"] for a in archs]

    x = np.arange(len(archs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_l = ax.bar(x - w/2, [v * 100 for v in r_lows],  w,
                    label="Low-freq gradient overlap", color="#2196F3", alpha=0.85)
    bars_h = ax.bar(x + w/2, [v * 100 for v in r_highs], w,
                    label="High-freq gradient overlap", color="#FF5722", alpha=0.85)

    ax.set_ylabel("Overlap ratio $r$ (%)\n"
                  r"$r = \|\mathbf{U}^\top \mathbf{g}\|^2 / \|\mathbf{g}\|^2$",
                  fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11)
    ax.set_title("Fisher subspace captures low-frequency gradients\n"
                 "(Cylinder, avg. over protected layers)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 在每个 bar 上标数值
    for bar in list(bars_l) + list(bars_h):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_pdf = OUT_DIR / "fig_gradient_overlap.pdf"
    out_png = OUT_DIR / "fig_gradient_overlap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n图片保存：{out_png}")

    # ── 打印数值表格 ──────────────────────────────────
    print(f"\n{'Arch':<12} {'r_low':>8} {'r_high':>8} {'ratio':>8}  n_layers")
    print("-" * 48)
    for a in archs:
        r = results[a]
        ratio = r["r_low"] / (r["r_high"] + 1e-12)
        print(f"{a:<12} {r['r_low']*100:>7.2f}% {r['r_high']*100:>7.2f}% "
              f"{ratio:>7.2f}x  {r['n_layers']}")

    print("\n结论：r_low / r_high > 1 → Fisher 子空间对齐低频梯度，即捕获物理知识。")


if __name__ == "__main__":
    main()
