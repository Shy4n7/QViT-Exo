"""Phase 2 — Attention map visualization for ExoplanetViT.

Generates the publishable 3×3 comparison figure:
    Row 1: 3 confirmed planets  — attention overlaid on RP channel
    Row 2: 3 false positives    — attention overlaid on RP channel

Attention is extracted from the last ViT block using the CLS→patch
attention slice, then averaged across heads and upsampled to the
original image size.

Usage
-----
python scripts/visualize_attention.py \\
    --checkpoint  models/vit/best_model.pt \\
    --csv         data/splits/val.csv \\
    --data-dir    data/processed \\
    --output      figures/attention_comparison.png

# Show interactively instead of saving
python scripts/visualize_attention.py \\
    --checkpoint models/vit/best_model.pt \\
    --csv data/splits/val.csv \\
    --data-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, InterpolationMode

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.vit_model import ExoplanetViT

_VIT_IMAGE_SIZE = 224
_PATCH_SIZE = 16
_N_PATCHES_SIDE = _VIT_IMAGE_SIZE // _PATCH_SIZE   # 14
_N_PATCHES = _N_PATCHES_SIDE ** 2                   # 196
_N_SAMPLES_PER_CLASS = 3


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class _ResizeTo224:
    def __init__(self) -> None:
        self._resize = Resize(
            (_VIT_IMAGE_SIZE, _VIT_IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


# ---------------------------------------------------------------------------
# Attention extraction helpers
# ---------------------------------------------------------------------------

def _cls_attention_last_block(
    attn_maps: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Extract the CLS→patch attention from the last block, averaged over heads.

    Parameters
    ----------
    attn_maps : dict[str, Tensor]
        Output of ExoplanetViT.get_attention_maps().
        Each value has shape (B, num_heads, seq_len, seq_len).

    Returns
    -------
    Tensor  shape (B, 14, 14)
        Spatial attention map in patch coordinates.
    """
    last_key = f"block_{max(int(k.split('_')[1]) for k in attn_maps)}"
    attn = attn_maps[last_key]           # (B, heads, 197, 197)

    # CLS token is position 0; patches are positions 1..196
    cls_attn = attn[:, :, 0, 1:]        # (B, heads, 196)
    cls_attn = cls_attn.mean(dim=1)      # (B, 196)  — average heads
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)  # normalise

    return cls_attn.reshape(-1, _N_PATCHES_SIDE, _N_PATCHES_SIDE)  # (B, 14, 14)


def _upsample_attention(
    attn: torch.Tensor,
    target_size: int,
) -> np.ndarray:
    """Bilinear upsample a (14, 14) attention map to (target_size, target_size).

    Returns
    -------
    np.ndarray  shape (target_size, target_size), dtype float32
    """
    attn_4d = attn.unsqueeze(0).unsqueeze(0)   # (1, 1, 14, 14)
    up = F.interpolate(
        attn_4d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return up.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_samples(
    dataset: ExoplanetDataset,
    n_per_class: int = _N_SAMPLES_PER_CLASS,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Return n_per_class images, aux tensors, and labels for each class."""
    images_out: list[torch.Tensor] = []
    aux_out: list[torch.Tensor] = []
    labels_out: list[int] = []

    counts = {0: 0, 1: 0}
    target_total = n_per_class * 2

    for image, aux, label in dataset:
        if counts[label] < n_per_class:
            images_out.append(image)
            aux_out.append(aux)
            labels_out.append(label)
            counts[label] += 1
        if sum(counts.values()) >= target_total:
            break

    # Sort: planets first (label=1), then FPs (label=0)
    combined = sorted(
        zip(images_out, aux_out, labels_out),
        key=lambda t: -t[2],
    )
    images_out, aux_out, labels_out = zip(*combined) if combined else ([], [], [])
    return list(images_out), list(aux_out), list(labels_out)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _make_attention_figure(
    images: list[torch.Tensor],
    aux_tensors: list[torch.Tensor],
    labels: list[int],
    model: ExoplanetViT,
    original_size: int = 64,
    save_path: str | None = None,
) -> None:
    """Create and optionally save the 2×3 attention overlay figure.

    Layout:
        Row 0 — Confirmed Planets  (label=1)
        Row 1 — False Positives    (label=0)
    Each column is one sample, showing:
        - Recurrence Plot channel (channel 0) as grayscale background
        - Quantum-attention heatmap overlaid with transparency
    """
    n_per_row = _N_SAMPLES_PER_CLASS
    fig, axes = plt.subplots(
        2, n_per_row,
        figsize=(4 * n_per_row, 9),
        constrained_layout=True,
    )

    row_titles = {1: "Confirmed Planet", 0: "False Positive"}
    row_colours = {1: "#2ECC71", 0: "#E74C3C"}

    planet_samples = [(img, aux, lbl) for img, aux, lbl in zip(images, aux_tensors, labels) if lbl == 1]
    fp_samples     = [(img, aux, lbl) for img, aux, lbl in zip(images, aux_tensors, labels) if lbl == 0]

    for row_idx, (row_samples, row_label) in enumerate(
        [(planet_samples, 1), (fp_samples, 0)]
    ):
        for col_idx, (img, aux, _) in enumerate(row_samples[:n_per_row]):
            ax = axes[row_idx][col_idx]

            # ── Extract attention map ─────────────────────────────
            img_batch = img.unsqueeze(0)       # (1, 2, 224, 224)
            aux_batch = aux.unsqueeze(0)       # (1, 5)
            attn_maps = model.get_attention_maps(img_batch, aux_batch)
            attn_14   = _cls_attention_last_block(attn_maps)[0]   # (14, 14)
            attn_up   = _upsample_attention(attn_14, target_size=_VIT_IMAGE_SIZE)

            # ── Original image (RP channel, channel 0) at 224×224 ─
            rp_channel = img[0].cpu().numpy()   # (224, 224)

            # ── Overlay ─────────────────────────────────────────────
            ax.imshow(rp_channel, cmap="gray", aspect="auto")
            overlay = ax.imshow(
                attn_up,
                cmap="hot",
                alpha=0.5,
                aspect="auto",
                vmin=attn_up.min(),
                vmax=attn_up.max(),
            )

            # ── Formatting ──────────────────────────────────────────
            if col_idx == 0:
                ax.set_ylabel(
                    row_titles[row_label],
                    fontsize=12,
                    fontweight="bold",
                    color=row_colours[row_label],
                )
            ax.set_title(
                f"Sample {col_idx + 1}",
                fontsize=10,
                color=row_colours[row_label],
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Colourbar on last column only
            if col_idx == n_per_row - 1:
                fig.colorbar(overlay, ax=ax, shrink=0.8, label="Attention")

    fig.suptitle(
        "Quantum ViT — Attention Maps: Confirmed Planets vs False Positives\n"
        "Channel: Recurrence Plot  |  Overlay: CLS→patch attention (last block, head-averaged)",
        fontsize=12,
        y=1.02,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Visualize ExoplanetViT attention maps"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model.pt saved by train_vit.py",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with kepoi_name + koi_disposition (val or test split)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory of processed samples",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save figure to this path (e.g. figures/attention.png). "
             "If omitted, displays interactively.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=_N_SAMPLES_PER_CLASS,
        help="Number of samples per class in the figure (default 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load model ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetViT(pretrained=False, regression_head=True)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Load dataset ─────────────────────────────────────────────────
    transform = _ResizeTo224()
    dataset = ExoplanetDataset(
        split_csv_path=args.csv,
        processed_dir=args.data_dir,
        transform=transform,
    )
    print(f"Dataset: {len(dataset)} samples")

    # ── Collect balanced samples ──────────────────────────────────────
    images, aux_tensors, labels = _collect_samples(dataset, n_per_class=args.n_per_class)
    planets = sum(1 for l in labels if l == 1)
    fps = sum(1 for l in labels if l == 0)
    print(f"Collected {planets} planets, {fps} false positives")

    if planets < args.n_per_class or fps < args.n_per_class:
        print(
            f"Warning: fewer than {args.n_per_class} samples in one class. "
            "Figure may have empty panels."
        )

    # Move tensors to device for inference
    images_device = [img.to(device) for img in images]
    aux_device = [aux.to(device) for aux in aux_tensors]

    # ── Generate figure ───────────────────────────────────────────────
    _make_attention_figure(
        images=images_device,
        aux_tensors=aux_device,
        labels=labels,
        model=model,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
