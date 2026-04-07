"""Phase 5 — Interpretability analysis pipeline.

Generates:
  1. figures/attention_lightcurve_3x3.png  — 3 planets / 3 FPs, attention on light curve
  2. figures/attention_comparison_3x3.png  — quantum vs classical attention overlay
  3. results/interpretability/stats_report.txt — correlation + MWU test results

Usage
-----
python scripts/run_interpretability.py \\
    --quantum-checkpoint models/quantum_vit/best_model.pt \\
    --classical-checkpoint models/vit/best_model.pt \\
    --val-csv data/splits/val.csv \\
    --data-dir data/processed
"""

from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, Resize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.vit_model import ExoplanetViT
from src.models.quantum_vit import ExoplanetQuantumViT
from src.interpretability.attention_analysis import (
    extract_attention_batch,
    attention_to_lightcurve_profile,
    cls_attention_last_block,
    upsample_to,
    ingress_egress_indicator,
    correlate_with_transit,
    compare_quantum_classical_attention,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


class _Resize224:
    def __init__(self):
        self._r = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
    def __call__(self, x):
        return self._r(x)


def _load_quantum(path: str, mode: str = "qonn_attn") -> ExoplanetQuantumViT:
    m = ExoplanetQuantumViT(quantum_mode=mode, pretrained=False, regression_head=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return m.eval()


def _load_classical(path: str) -> ExoplanetViT:
    m = ExoplanetViT(pretrained=False, regression_head=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return m.eval()


# ---------------------------------------------------------------------------
# Figure 1: attention overlaid on light curve (1-D profile)
# ---------------------------------------------------------------------------

def _plot_attention_lightcurve(
    data: dict[str, list],
    save_path: str,
    n_per_class: int = 3,
) -> None:
    """2-row × 3-col: top row planets, bottom row FPs.
    Each panel: light curve (RP row-sum proxy) + attention profile overlay.
    """
    planets = [i for i, l in enumerate(data["labels"]) if l == 1][:n_per_class]
    fps     = [i for i, l in enumerate(data["labels"]) if l == 0][:n_per_class]

    fig, axes = plt.subplots(2, n_per_class, figsize=(4 * n_per_class, 8),
                             constrained_layout=True)
    colours  = {1: "#2ECC71", 0: "#E74C3C"}
    row_lbls = {1: "Confirmed Planet", 0: "False Positive"}

    for row, (indices, label) in enumerate([(planets, 1), (fps, 0)]):
        for col, idx in enumerate(indices):
            ax   = axes[row][col]
            a2d  = data["attention_2d"][idx]       # (14, 14)
            prof = data["profile_1d"][idx]         # (14,)
            t    = np.linspace(0, 1, len(prof))

            # RP row-sum as a proxy for phase-folded flux shape
            rp_proxy = a2d.sum(axis=1)
            rp_proxy = (rp_proxy - rp_proxy.min()) / (rp_proxy.max() - rp_proxy.min() + 1e-8)

            ax2 = ax.twinx()
            ax.fill_between(t, prof, alpha=0.4, color=colours[label], label="Attention")
            ax.plot(t, prof, color=colours[label], lw=1.5)
            ax2.plot(t, rp_proxy, "k--", lw=1.2, alpha=0.6, label="RP proxy")

            ax.set_xlabel("Normalised time", fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbls[label], fontsize=10,
                              fontweight="bold", color=colours[label])
            ax.set_title(f"Sample {col+1}", fontsize=9, color=colours[label])
            ax.set_yticks([]); ax2.set_yticks([])

    fig.suptitle(
        "Quantum ViT Attention Profile vs Light Curve\n"
        "1-D attention (coloured) | RP row-sum proxy (dashed)",
        fontsize=12,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Light-curve figure saved → %s", save_path)


# ---------------------------------------------------------------------------
# Figure 2: quantum vs classical attention side-by-side
# ---------------------------------------------------------------------------

def _plot_quantum_vs_classical(
    q_data: dict, c_data: dict,
    save_path: str,
    n_per_class: int = 3,
) -> None:
    """3 columns per class, each column: quantum | classical attention maps."""
    planets = [i for i, l in enumerate(q_data["labels"]) if l == 1][:n_per_class]
    fps     = [i for i, l in enumerate(q_data["labels"]) if l == 0][:n_per_class]

    n_cols  = n_per_class * 2   # quantum + classical per sample
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8),
                             constrained_layout=True)
    colours   = {1: "#2ECC71", 0: "#E74C3C"}
    row_lbls  = {1: "Planet", 0: "False Positive"}

    for row, (indices, label) in enumerate([(planets, 1), (fps, 0)]):
        for s, idx in enumerate(indices):
            q_map = upsample_to(torch.tensor(q_data["attention_2d"][idx]), 14)
            c_map = upsample_to(torch.tensor(c_data["attention_2d"][idx]), 14)

            col_q = s * 2
            col_c = s * 2 + 1

            axes[row][col_q].imshow(q_map, cmap="hot", aspect="auto")
            axes[row][col_q].set_title(f"{row_lbls[label]} {s+1}\nQuantum",
                                       fontsize=8, color=colours[label])
            axes[row][col_q].axis("off")

            axes[row][col_c].imshow(c_map, cmap="hot", aspect="auto")
            axes[row][col_c].set_title(f"{row_lbls[label]} {s+1}\nClassical",
                                       fontsize=8, color="gray")
            axes[row][col_c].axis("off")

    fig.suptitle(
        "Quantum vs Classical ViT — CLS Attention Maps\n"
        "Last block, head-averaged",
        fontsize=12,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison figure saved → %s", save_path)


# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------

def _run_statistics(q_data: dict, c_data: dict) -> str:
    lines = ["=" * 60, "  PHASE 5 — INTERPRETABILITY STATISTICS", "=" * 60]

    planet_idx = [i for i, l in enumerate(q_data["labels"]) if l == 1]
    fp_idx     = [i for i, l in enumerate(q_data["labels"]) if l == 0]

    # --- Ingress/egress correlation (planets only) ---
    lines.append("\n  Ingress/Egress Correlation (confirmed planets)")
    lines.append("  " + "-" * 40)
    corr_results = []
    for idx in planet_idx[:20]:  # first 20 planets
        prof = q_data["profile_1d"][idx]
        n    = len(prof)
        # Synthetic transit params (mid=0.5, duration=0.15) — real values
        # would come from KOI catalog; this demonstrates the method
        ind  = ingress_egress_indicator(n, t_mid=0.5, duration=0.15)
        r    = correlate_with_transit(prof, ind, n_permutations=200)
        corr_results.append(r)

    sig   = sum(1 for r in corr_results if r.significant)
    rhos  = [r.spearman_rho for r in corr_results]
    pvals = [r.permutation_p for r in corr_results]
    lines.append(f"  Planets analysed     : {len(corr_results)}")
    lines.append(f"  Significant (p<0.05) : {sig}/{len(corr_results)}")
    lines.append(f"  Mean Spearman ρ      : {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")
    lines.append(f"  Mean permutation-p   : {np.mean(pvals):.4f}")

    # --- Quantum vs classical MWU test ---
    lines.append("\n  Quantum vs Classical Attention (Mann-Whitney U)")
    lines.append("  " + "-" * 40)
    mwu_results = []
    common_idx = list(range(min(len(q_data["labels"]), len(c_data["labels"]), 40)))
    for idx in common_idx:
        r = compare_quantum_classical_attention(
            q_data["attention_2d"][idx],
            c_data["attention_2d"][idx],
        )
        mwu_results.append(r)

    sig_mwu = sum(1 for r in mwu_results if r.significant)
    q_ents  = [r.quantum_entropy for r in mwu_results]
    c_ents  = [r.classical_entropy for r in mwu_results]
    lines.append(f"  Samples analysed     : {len(mwu_results)}")
    lines.append(f"  Sig. different       : {sig_mwu}/{len(mwu_results)} (p<0.05)")
    lines.append(f"  Quantum entropy      : {np.mean(q_ents):.4f} ± {np.std(q_ents):.4f}")
    lines.append(f"  Classical entropy    : {np.mean(c_ents):.4f} ± {np.std(c_ents):.4f}")
    lines.append(
        f"  Interpretation       : quantum attention is "
        + ("MORE DIFFUSE" if np.mean(q_ents) > np.mean(c_ents) else "MORE FOCUSED")
        + " than classical"
    )
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — Interpretability analysis")
    p.add_argument("--quantum-checkpoint",   required=True)
    p.add_argument("--classical-checkpoint", required=True)
    p.add_argument("--quantum-mode", default="qonn_attn",
                   choices=["vqc_head", "qonn_attn"])
    p.add_argument("--val-csv",    required=True)
    p.add_argument("--data-dir",   required=True)
    p.add_argument("--figures-dir", default="figures/")
    p.add_argument("--results-dir", default="results/interpretability/")
    p.add_argument("--max-samples", type=int, default=200)
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figs   = Path(args.figures_dir)
    res    = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models…")
    q_model = _load_quantum(args.quantum_checkpoint, args.quantum_mode)
    c_model = _load_classical(args.classical_checkpoint)

    ds = ExoplanetDataset(args.val_csv, args.data_dir, transform=_Resize224(), skip_missing=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    logger.info("Extracting attention maps (max %d samples)…", args.max_samples)
    q_data = extract_attention_batch(q_model, loader, device, args.max_samples)
    c_data = extract_attention_batch(c_model, loader, device, args.max_samples)
    logger.info("Quantum samples: %d | Classical samples: %d",
                len(q_data["labels"]), len(c_data["labels"]))

    logger.info("Generating figures…")
    _plot_attention_lightcurve(q_data, str(figs / "attention_lightcurve_3x3.png"))
    _plot_quantum_vs_classical(q_data, c_data, str(figs / "attention_comparison_3x3.png"))

    logger.info("Running statistical tests…")
    report = _run_statistics(q_data, c_data)
    logger.info("\n%s", report)
    (res / "stats_report.txt").write_text(report, encoding="utf-8")
    logger.info("Stats report saved → %s", res / "stats_report.txt")

    # Save raw attention profiles as numpy for further analysis
    np.save(str(res / "quantum_profiles.npy"),
            np.array(q_data["profile_1d"]))
    np.save(str(res / "classical_profiles.npy"),
            np.array(c_data["profile_1d"]))
    np.save(str(res / "labels.npy"),
            np.array(q_data["labels"]))
    logger.info("Phase 5 complete.")


if __name__ == "__main__":
    main()
