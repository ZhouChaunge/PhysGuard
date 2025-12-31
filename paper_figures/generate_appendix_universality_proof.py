#!/usr/bin/env python3
"""
Generate NeurIPS-proof-style appendix figures for ALL datasets.

For each dataset, output one figure with architecture columns:
  Row 1: radial PSD of delta-y (FIM top-1 vs random)
  Row 2: violin (FIM top-K eigenvectors vs random directions)

Outputs:
  figures/appendix_universality_proof_<dataset>.png
  figures/appendix_universality_proof_<dataset>.pdf
"""

import os
import re
import sys
import glob
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..", "RealPDEBench")
sys.path.insert(0, REPO_DIR)

ROOT = "."
RESULTS_ROOT = f"{ROOT}/RealPDEBench/results"
DATASET_ROOT = f"{ROOT}/data/realpdebench"
FIG_DIR = f"{ROOT}/figures"
CACHE_DIR = "/tmp/appendix_neurips_style_cache"

ARCH_ORDER = ["FNO", "CNO", "DeepONet", "Transolver"]

DATASETS = {
    "cylinder": {
        "result_dir": "001-cylinder",
        "dataset_name": "cylinder",
        "ds_import": "realpdebench.data.fluid_hf_dataset",
        "ds_class": "CylinderHFDataset",
        "title": "Cylinder",
        "fno_modes2": 12,
    },
    "controlled_cylinder": {
        "result_dir": "002-control_cylinder",
        "dataset_name": "controlled_cylinder",
        "ds_import": "realpdebench.data.fluid_hf_dataset",
        "ds_class": "ControlledCylinderHFDataset",
        "title": "Controlled Cylinder",
        "fno_modes2": 12,
    },
    "combustion": {
        "result_dir": "003-combustion",
        "dataset_name": "combustion",
        "ds_import": "realpdebench.data.combustion_hf_dataset",
        "ds_class": "CombustionHFDataset",
        "title": "Combustion",
        "fno_modes2": 16,
    },
}

# Keep these moderate so running all datasets is practical.
N_FIM = 24
N_TEST = 20
K_VIOLIN = 12
N_RANDOM_PSD = 10
EPSILON = 1e-3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def arch_build_kwargs(arch: str, ds_key: str) -> Dict:
    modes2 = DATASETS[ds_key]["fno_modes2"]
    if arch == "FNO":
        return dict(model_name="fno", modes1=4, modes2=modes2, modes3=16,
                    n_layers=4, width=64)
    if arch == "CNO":
        return dict(model_name="cno", N_layers=3)
    if arch == "DeepONet":
        return dict(model_name="deeponet", p=128, dropout_rate=0.1)
    if arch == "Transolver":
        return dict(
            model_name="transolver",
            space_dim=3, n_layers=1, n_hidden=256, n_head=8,
            H=128, W=64, D=20,
            fun_dim=0, out_dim=3, ref=4,
            dropout=0.1, act="gelu", mlp_ratio=4, slice_num=16,
        )
    raise ValueError(f"Unknown architecture: {arch}")


def arch_fwd_batch(arch: str) -> int:
    return 1 if arch == "Transolver" else 4


def fixed_target_param(arch: str):
    if arch == "FNO":
        return "spectral_convs.1.weights1"
    if arch == "DeepONet":
        return "trunk.fc.4.weight"
    return None


def resolve_ckpt_path(ds_key: str, arch: str) -> str:
    ds = DATASETS[ds_key]
    arch_dir = arch.lower() if arch != "DeepONet" else "deeponet"
    base = os.path.join(RESULTS_ROOT, ds["result_dir"], arch_dir)
    pre_dirs = sorted(glob.glob(os.path.join(base, "*pretrained")))
    if not pre_dirs:
        raise FileNotFoundError(f"No pretrained dir for {ds_key}/{arch} under {base}")

    run_dirs = []
    for p in pre_dirs:
        run_dirs.extend(glob.glob(os.path.join(p, "20*")))
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise FileNotFoundError(f"No timestamp run dir for {ds_key}/{arch}")

    latest_run = run_dirs[-1]
    model_paths = sorted(glob.glob(os.path.join(latest_run, "model_*.pth")))
    if not model_paths:
        raise FileNotFoundError(f"No model_*.pth under {latest_run}")

    def step_num(path: str) -> int:
        m = re.search(r"model_(\d+)\.pth$", path)
        return int(m.group(1)) if m else -1

    model_paths.sort(key=step_num)
    return model_paths[-1]


def measure_f_low(psd: np.ndarray, half: int) -> float:
    kth = max(1, round(half / 3))
    return float(psd[:kth].sum() / (psd.sum() + 1e-15))


def compute_2d_psd(dy: torch.Tensor):
    b, t, h, w, c = dy.shape
    half = min(h // 2, w // 2)

    dy_avg = dy.float().mean(dim=1)
    ff = torch.fft.fftn(dy_avg, dim=[1, 2])
    pp = ff.abs() ** 2
    pp_half = pp[:, :h // 2, :w // 2, :]

    ii = torch.arange(h // 2).float()
    jj = torch.arange(w // 2).float()
    gi, gj = torch.meshgrid(ii, jj, indexing="ij")
    radial = torch.floor(torch.sqrt(gi ** 2 + gj ** 2)).long()
    valid = radial < half

    rflat = radial[valid]
    pbc = pp_half.permute(0, 3, 1, 2).reshape(b * c, h // 2, w // 2)
    psum = pbc.sum(dim=0)
    pvalid = psum[valid]

    psd = torch.zeros(half)
    psd.scatter_add_(0, rflat, pvalid)
    cnt = torch.zeros(half, dtype=torch.long)
    cnt.scatter_add_(0, rflat, torch.ones_like(rflat))
    psd = psd / (cnt.clamp(min=1).float() * b * c)
    return psd.numpy(), half


def build_data_and_normalizer(ds_key: str):
    import importlib
    from realpdebench.data.data_normalizer import GaussianNormalizer

    cfg = DATASETS[ds_key]
    mod = importlib.import_module(cfg["ds_import"])
    ds_cls = getattr(mod, cfg["ds_class"])

    common = dict(dataset_name=cfg["dataset_name"], dataset_root=DATASET_ROOT)
    train_ds = ds_cls(mode="train", dataset_type="numerical", **common)
    val_ds = ds_cls(mode="val", dataset_type="real", **common)
    normalizer = GaussianNormalizer(train_ds, device=DEVICE)
    return train_ds, val_ds, normalizer


def build_and_load_model(arch: str, ds_key: str, train_ds):
    from realpdebench.model.load_model import load_model

    ckpt_path = resolve_ckpt_path(ds_key, arch)
    log.info(f"[{ds_key}/{arch}] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    raw = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in raw.items()}

    kwargs = arch_build_kwargs(arch, ds_key)

    # Transolver H/W/D should match current dataset shape.
    if arch == "Transolver":
        sample_x, _ = train_ds[0]
        # sample_x shape: [T, H, W, C]; Transolver config uses H=W_dim, W=H_dim.
        kwargs["D"] = int(sample_x.shape[0])
        kwargs["H"] = int(sample_x.shape[2])
        kwargs["W"] = int(sample_x.shape[1])

    # DeepONet width differs across datasets; infer p from checkpoint if needed.
    if arch == "DeepONet" and "trunk.fc.4.weight" in state:
        inferred_p = int(state["trunk.fc.4.weight"].shape[0])
        kwargs["p"] = inferred_p
        log.info(f"[{ds_key}/{arch}] inferred p={inferred_p} from checkpoint")

    # Transolver input dimension differs across datasets (e.g., controlled cylinder).
    if arch == "Transolver" and "preprocess.linear_pre.0.weight" in state:
        inferred_space_dim = int(state["preprocess.linear_pre.0.weight"].shape[1])
        kwargs["space_dim"] = inferred_space_dim
        log.info(f"[{ds_key}/{arch}] inferred space_dim={inferred_space_dim} from checkpoint")
    if arch == "Transolver" and "blocks.0.mlp2.weight" in state:
        inferred_out_dim = int(state["blocks.0.mlp2.weight"].shape[0])
        kwargs["out_dim"] = inferred_out_dim
        log.info(f"[{ds_key}/{arch}] inferred out_dim={inferred_out_dim} from checkpoint")

    model = load_model(train_ds, device=DEVICE, **kwargs)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def real_numel(p):
    return 2 * p.numel() if p.is_complex() else p.numel()


def pick_target_param(model, arch: str):
    fixed = fixed_target_param(arch)
    if fixed is not None:
        for n, _ in model.named_parameters():
            if n == fixed:
                return fixed

    cand = []
    for n, p in model.named_parameters():
        if not p.requires_grad or p.dim() < 2:
            continue
        low = n.lower()
        if any(tok in low for tok in ("bias", "norm", ".bn", "embed", "placeholder")):
            continue
        cand.append((n, real_numel(p)))
    cand.sort(key=lambda x: -x[1])
    return cand[0][0]


def collect_grads(model, loader, normalizer, target_name, n):
    grads = []
    cnt = 0
    pbar = tqdm(total=n, desc="  FIM grads", leave=False)
    for x, y in loader:
        if cnt >= n:
            break
        x, _ = normalizer.preprocess(x, y)
        x = x[:1].to(DEVICE)
        model.zero_grad()
        model(x).pow(2).mean().backward()
        for pname, p in model.named_parameters():
            if pname != target_name:
                continue
            if p.grad is None:
                break
            if p.is_complex():
                gv = torch.cat([p.grad.real.reshape(-1), p.grad.imag.reshape(-1)]).float().cpu()
            else:
                gv = p.grad.reshape(-1).float().cpu()
            grads.append(gv)
            break
        cnt += 1
        pbar.update(1)
    pbar.close()
    return grads


def top_k_eigvec(grads, k):
    jmat = torch.stack(grads, dim=0)
    gram = jmat @ jmat.T
    lam, vec = np.linalg.eigh(gram.numpy().astype(np.float64))
    lam = np.maximum(lam[::-1], 0.0)
    vec = vec[:, ::-1]

    valid = int((lam > 1e-12 * (lam[0] + 1e-30)).sum())
    use_k = min(k, max(1, valid))

    u_list = []
    for j in range(use_k):
        vj = torch.tensor(vec[:, j], dtype=torch.float32)
        uj = jmat.T @ vj / (lam[j] ** 0.5 + 1e-12)
        uj = uj / (uj.norm() + 1e-12)
        u_list.append(uj)
    return torch.stack(u_list, dim=1)


def get_base_outputs(model, loader, normalizer, n):
    xs, ys = [], []
    cnt = 0
    with torch.no_grad():
        for x, y in loader:
            if cnt >= n:
                break
            x, _ = normalizer.preprocess(x, y)
            rem = min(n - cnt, x.shape[0])
            xx = x[:rem]
            xs.append(xx.cpu())
            ys.append(model(xx.to(DEVICE)).cpu())
            cnt += rem
    return torch.cat(xs), torch.cat(ys)


def perturb_psd(model, target_name, u_vec, x_test, y_base, fwd_batch):
    for pname, p in model.named_parameters():
        if pname != target_name:
            continue

        if p.is_complex():
            d = p.numel()
            re = u_vec[:d].reshape(p.shape).to(p.device)
            im = u_vec[d:].reshape(p.shape).to(p.device)
            delta = torch.complex(re, im).to(dtype=p.dtype)
        else:
            delta = u_vec.reshape(p.shape).to(device=p.device, dtype=p.dtype)

        with torch.no_grad():
            p.data.add_(EPSILON * delta)

        ys = []
        with torch.no_grad():
            for st in range(0, x_test.shape[0], fwd_batch):
                ys.append(model(x_test[st:st + fwd_batch].to(DEVICE)).cpu())
        y_pert = torch.cat(ys)

        with torch.no_grad():
            p.data.sub_(EPSILON * delta)

        dy = y_pert - y_base
        psd, half = compute_2d_psd(dy)
        return psd, half

    raise ValueError(f"Target parameter not found: {target_name}")


def compute_metrics_for_arch(ds_key: str, arch: str, train_ds, val_ds, normalizer):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{ds_key}_{arch}.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        return {
            "psd_rank1": d["psd_rank1"],
            "psd_random": d["psd_random"],
            "half": int(d["half"]),
            "fl_top": d["fl_top"],
            "fl_random": d["fl_random"],
            "pval": float(d["pval"]),
        }

    log.info(f"[{ds_key}/{arch}] computing metrics ...")
    model = build_and_load_model(arch, ds_key, train_ds)
    target = pick_target_param(model, arch)
    fwd_batch = arch_fwd_batch(arch)

    loader_fim = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    loader_test = DataLoader(val_ds, batch_size=fwd_batch, shuffle=False, num_workers=0, pin_memory=True)

    grads = collect_grads(model, loader_fim, normalizer, target, N_FIM)
    u_top = top_k_eigvec(grads, K_VIOLIN)
    x_test, y_base = get_base_outputs(model, loader_test, normalizer, N_TEST)

    # Row 1
    u1 = u_top[:, 0]
    psd_rank1, half = perturb_psd(model, target, u1, x_test, y_base, fwd_batch)

    psd_rand_list = []
    for i in range(N_RANDOM_PSD):
        torch.manual_seed(SEED + 1000 + i)
        r = torch.randn(u_top.shape[0])
        r = r / (r.norm() + 1e-12)
        pr, _ = perturb_psd(model, target, r, x_test, y_base, fwd_batch)
        psd_rand_list.append(pr)
    psd_random = np.mean(psd_rand_list, axis=0)

    # Row 2
    fl_top = []
    for j in range(u_top.shape[1]):
        p_j, h_j = perturb_psd(model, target, u_top[:, j], x_test, y_base, fwd_batch)
        fl_top.append(measure_f_low(p_j, h_j))

    fl_rand = []
    for i in range(u_top.shape[1]):
        torch.manual_seed(SEED + 3000 + i)
        r = torch.randn(u_top.shape[0])
        r = r / (r.norm() + 1e-12)
        p_r, h_r = perturb_psd(model, target, r, x_test, y_base, fwd_batch)
        fl_rand.append(measure_f_low(p_r, h_r))

    fl_top = np.array(fl_top, dtype=np.float64)
    fl_rand = np.array(fl_rand, dtype=np.float64)
    pval = float(mannwhitneyu(fl_top, fl_rand, alternative="greater").pvalue)

    out = {
        "psd_rank1": psd_rank1,
        "psd_random": psd_random,
        "half": half,
        "fl_top": fl_top,
        "fl_random": fl_rand,
        "pval": pval,
    }
    np.savez(cache, **out)

    del model
    torch.cuda.empty_cache()
    return out


def make_figure(ds_key: str, all_arch_data: Dict[str, Dict]):
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig = plt.figure(figsize=(7.4, 3.6))
    gs = GridSpec(2, len(ARCH_ORDER), hspace=0.62, wspace=0.35,
                  left=0.07, right=0.995, top=0.91, bottom=0.12)

    c_fim = "#1565C0"
    c_rand = "#C62828"
    c_band = "#90CAF9"

    for col, arch in enumerate(ARCH_ORDER):
        res = all_arch_data[arch]
        half = int(res["half"])
        k_th = max(1, round(half / 3))
        ks = np.arange(half)

        ax = fig.add_subplot(gs[0, col])
        psd1 = res["psd_rank1"]
        psdr = res["psd_random"]
        fl1 = measure_f_low(psd1, half)
        flr = measure_f_low(psdr, half)
        psd1n = psd1 / (psd1.sum() + 1e-15)
        psdrn = psdr / (psdr.sum() + 1e-15)

        ax.plot(ks, psd1n, color=c_fim, lw=1.6,
                label=rf"FIM top-1 ($f_{{\rm low}}={fl1:.3f}$)")
        ax.plot(ks, psdrn, color=c_rand, lw=1.3, ls="-.",
                label=rf"Random ($f_{{\rm low}}={flr:.3f}$)")
        ax.axvspan(0, k_th - 0.5, alpha=0.10, color=c_band)
        ax.axvline(k_th - 0.5, color=c_band, lw=0.8, ls=":", alpha=0.7)
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("Energy density (norm.)")
        ax.set_xlabel("Radial wavenumber $k$")
        ax.set_title(arch, fontsize=8, pad=3)
        ax.legend(loc="best", framealpha=0.92, borderpad=0.3,
                  handlelength=1.0, fontsize=5.5)
        ax.set_xlim(0, half - 1)
        ax.set_ylim(bottom=1e-6)
        ax.grid(axis="y", alpha=0.2, lw=0.4)

        ax2 = fig.add_subplot(gs[1, col])
        fl_fim = res["fl_top"]
        fl_rand = res["fl_random"]

        parts = ax2.violinplot([fl_fim, fl_rand], positions=[1, 2], widths=0.6,
                               showmeans=True, showmedians=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            cc = c_fim if i == 0 else c_rand
            body.set_facecolor(cc)
            body.set_alpha(0.25)
            body.set_edgecolor(cc)
            body.set_linewidth(0.8)
        parts["cmeans"].set_colors([c_fim, c_rand])
        parts["cmeans"].set_linewidths(1.5)

        rng = np.random.default_rng(SEED + col)
        j1 = 1 + rng.uniform(-0.12, 0.12, len(fl_fim))
        j2 = 2 + rng.uniform(-0.12, 0.12, len(fl_rand))
        ax2.scatter(j1, fl_fim, color=c_fim, s=4, alpha=0.5, edgecolors="none", zorder=3)
        ax2.scatter(j2, fl_rand, color=c_rand, s=8, alpha=0.6, edgecolors="none", zorder=3)

        pval = res["pval"]
        delta = float(fl_fim.mean() - fl_rand.mean())
        ax2.text(0.98, 0.95, f"$\\Delta={delta:+.3f}$\n$p$={pval:.1e}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=6.8)

        vals = np.concatenate([fl_fim, fl_rand])
        ylo = max(0.0, vals.min() - 0.03)
        yhi = min(1.05, vals.max() + 0.04)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["FIM\neigenvec.", "Random\ndir."], fontsize=6)
        if col == 0:
            ax2.set_ylabel("$f_{\\rm low}$")
        ax2.set_ylim(ylo, yhi)
        ax2.set_xlim(0.3, 2.7)
        ax2.grid(axis="y", alpha=0.2, lw=0.4)

    ds_title = DATASETS[ds_key]["title"]
    fig.suptitle(f"Appendix: NeurIPS-proof-style cross-architecture evidence ({ds_title})",
                 fontsize=9.8, y=0.98)

    out_base = os.path.join(FIG_DIR, f"appendix_universality_proof_{ds_key}")
    for ext in ("png", "pdf"):
        out = f"{out_base}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log.info(f"Saved -> {out}")
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for ds_key in DATASETS:
        log.info("=" * 72)
        log.info(f"Dataset: {ds_key}")
        train_ds, val_ds, normalizer = build_data_and_normalizer(ds_key)

        results = {}
        for arch in ARCH_ORDER:
            results[arch] = compute_metrics_for_arch(ds_key, arch, train_ds, val_ds, normalizer)
            fl_top = results[arch]["fl_top"]
            fl_rand = results[arch]["fl_random"]
            log.info(
                f"[{ds_key}/{arch}] FIM={fl_top.mean():.4f}  Rand={fl_rand.mean():.4f}  "
                f"Delta={fl_top.mean()-fl_rand.mean():+.4f}  p={results[arch]['pval']:.2e}"
            )

        make_figure(ds_key, results)

    log.info("All datasets finished.")


if __name__ == "__main__":
    main()
