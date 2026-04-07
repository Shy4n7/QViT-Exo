"""Phase 5 — Interpretability analysis for quantum ViT attention maps.

Three scientific questions answered here:

1. Light-curve overlay
   Recurrence plots (RP) encode time on both axes. A cell (i,j) in the RP
   corresponds to time steps t_i and t_j. The CLS→patch attention weight for
   that cell tells us how much the model attended to that time-pair.
   Marginalising over one axis gives a 1-D attention profile over time, which
   can be overlaid on the original flux light curve.

2. Ingress/egress correlation
   For confirmed planets we know the transit mid-time t_mid, duration T14,
   and depth δ from the KOI catalog. Ingress is [t_mid - T14/2, t_mid - T14/4]
   and egress is [t_mid + T14/4, t_mid + T14/2].
   We compute Spearman ρ between the 1-D attention profile and a binary
   ingress+egress indicator vector, then test against a permutation null.

3. Quantum vs classical attention difference
   For the same sample, compare the last-block CLS attention from the quantum
   ViT (QONN path) and the classical ViT. Statistical test: two-sample
   Mann-Whitney U on the distribution of attention weights. A significant
   p-value means quantum attention genuinely differs — publishable result
   either way (yes or no).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Attention extraction helpers (mirrors visualize_attention.py)
# ---------------------------------------------------------------------------

_N_PATCHES_SIDE = 14   # 224 / 16 for ViT-B/16
_N_PATCHES      = _N_PATCHES_SIDE ** 2   # 196


def cls_attention_last_block(
    attn_maps: dict[str, torch.Tensor],
) -> torch.Tensor:
    """CLS→patch attention from last block, head-averaged.

    Returns
    -------
    Tensor (B, 14, 14)  Spatial attention in patch coordinates, normalised.
    """
    last = f"block_{max(int(k.split('_')[1]) for k in attn_maps)}"
    attn = attn_maps[last]                     # (B, heads, 197, 197)
    cls  = attn[:, :, 0, 1:].mean(dim=1)       # (B, 196)
    cls  = cls / (cls.sum(dim=-1, keepdim=True) + 1e-8)
    return cls.reshape(-1, _N_PATCHES_SIDE, _N_PATCHES_SIDE)


def upsample_to(attn: torch.Tensor, size: int) -> np.ndarray:
    """Bilinear upsample a (14,14) map to (size, size). Returns numpy."""
    up = F.interpolate(
        attn.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size), mode="bilinear", align_corners=False,
    )
    return up.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# 2D attention → 1D light-curve profile
# ---------------------------------------------------------------------------

def attention_to_lightcurve_profile(
    attn_2d: np.ndarray,
) -> np.ndarray:
    """Collapse a 2-D attention map to a 1-D time profile.

    A recurrence plot cell (i, j) represents the pair of time steps
    (t_i, t_j).  The attention weight at (i, j) indicates how much the
    model considered that time-pair.  Summing over j gives the total
    attention drawn to each time step t_i.

    Parameters
    ----------
    attn_2d : ndarray (H, W)  Attention map (square, H == W == time steps).

    Returns
    -------
    ndarray (H,)  1-D attention profile, normalised to sum=1.
    """
    profile = attn_2d.sum(axis=1)          # sum over columns → (H,)
    total   = profile.sum()
    if total > 0:
        profile = profile / total
    return profile


# ---------------------------------------------------------------------------
# Ingress / egress correlation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationResult:
    spearman_rho:   float
    p_value:        float
    permutation_p:  float   # empirical p from permutation test
    n_time_steps:   int
    significant:    bool    # p_value < 0.05


def ingress_egress_indicator(
    n_steps:  int,
    t_mid:    float,     # normalised mid-transit time in [0, 1]
    duration: float,     # normalised transit duration in [0, 1]
    pad:      float = 0.25,
) -> np.ndarray:
    """Binary indicator: 1 at ingress/egress windows, 0 elsewhere.

    Parameters
    ----------
    n_steps  : int    Number of time steps (same as attention profile length).
    t_mid    : float  Normalised mid-transit time [0, 1].
    duration : float  Normalised transit duration [0, 1].
    pad      : float  Fraction of half-duration defining ingress/egress width.

    Returns
    -------
    ndarray (n_steps,) bool cast to float.
    """
    t = np.linspace(0, 1, n_steps)
    half = duration / 2.0
    ingress_lo = t_mid - half
    ingress_hi = t_mid - half * (1 - pad)
    egress_lo  = t_mid + half * (1 - pad)
    egress_hi  = t_mid + half

    indicator = (
        ((t >= ingress_lo) & (t <= ingress_hi)) |
        ((t >= egress_lo)  & (t <= egress_hi))
    ).astype(float)
    return indicator


def correlate_with_transit(
    attention_profile: np.ndarray,
    indicator:         np.ndarray,
    n_permutations:    int = 1000,
    seed:              int = 42,
) -> CorrelationResult:
    """Spearman correlation between attention profile and ingress/egress indicator.

    Permutation test: shuffle the indicator ``n_permutations`` times and
    count how often the permuted |ρ| exceeds the observed |ρ| to get an
    empirical p-value free of distributional assumptions.

    Parameters
    ----------
    attention_profile : ndarray (T,)
    indicator         : ndarray (T,)  Binary ingress/egress mask.
    n_permutations    : int
    seed              : int

    Returns
    -------
    CorrelationResult
    """
    rho, pval = stats.spearmanr(attention_profile, indicator)
    rho = 0.0 if np.isnan(rho) else float(rho)
    pval = 1.0 if np.isnan(pval) else float(pval)

    rng = np.random.default_rng(seed)
    perm_rhos = np.array([
        abs(v) if not np.isnan(v) else 0.0
        for v in (
            stats.spearmanr(attention_profile, rng.permutation(indicator))[0]
            for _ in range(n_permutations)
        )
    ])
    perm_p = float((perm_rhos >= abs(rho)).mean())

    return CorrelationResult(
        spearman_rho  = rho,
        p_value       = pval,
        permutation_p = perm_p,
        n_time_steps  = len(attention_profile),
        significant   = pval < 0.05,
    )


# ---------------------------------------------------------------------------
# Quantum vs classical attention comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionComparisonResult:
    mannwhitney_u:    float
    p_value:          float
    significant:      bool     # p < 0.05 → quantum attention differs from classical
    quantum_mean:     float
    classical_mean:   float
    quantum_entropy:  float    # entropy of quantum attention distribution
    classical_entropy: float


def compare_quantum_classical_attention(
    quantum_attn:   np.ndarray,   # (14, 14) or (H, W) — quantum model
    classical_attn: np.ndarray,   # same shape — classical model
) -> AttentionComparisonResult:
    """Two-sample Mann-Whitney U test on flattened attention weight distributions.

    Tests H0: quantum and classical attention weights are drawn from the same
    distribution. Rejection → quantum attention is statistically different.
    Either result is publishable: difference validates quantum contribution;
    similarity suggests classical is sufficient (honest negative result).

    Parameters
    ----------
    quantum_attn   : ndarray  Attention map from quantum ViT.
    classical_attn : ndarray  Attention map from classical ViT.

    Returns
    -------
    AttentionComparisonResult
    """
    q_flat = quantum_attn.flatten()
    c_flat = classical_attn.flatten()

    u_stat, p_val = stats.mannwhitneyu(q_flat, c_flat, alternative="two-sided")

    def _entropy(x: np.ndarray) -> float:
        x = x / (x.sum() + 1e-12)
        return float(-np.sum(x * np.log(x + 1e-12)))

    return AttentionComparisonResult(
        mannwhitney_u     = float(u_stat),
        p_value           = float(p_val),
        significant       = float(p_val) < 0.05,
        quantum_mean      = float(q_flat.mean()),
        classical_mean    = float(c_flat.mean()),
        quantum_entropy   = _entropy(q_flat),
        classical_entropy = _entropy(c_flat),
    )


# ---------------------------------------------------------------------------
# Batch extraction over a DataLoader
# ---------------------------------------------------------------------------

def extract_attention_batch(
    model:       torch.nn.Module,
    loader,
    device:      torch.device,
    max_samples: int = 200,
) -> dict[str, list]:
    """Extract attention profiles + labels for all samples in loader.

    Returns
    -------
    dict with keys:
        "attention_2d"  : list of ndarray (14, 14)  — per-sample attention map
        "profile_1d"    : list of ndarray (14,)     — 1-D light-curve profile
        "labels"        : list of int               — 0=FP, 1=planet
        "logits"        : list of ndarray (2,)      — model output logits
    """
    model.eval()
    model.to(device)

    results: dict[str, list] = {
        "attention_2d": [], "profile_1d": [], "labels": [], "logits": []
    }
    collected = 0

    for image, aux, labels in loader:
        if collected >= max_samples:
            break
        image = image.to(device)
        aux   = aux.to(device)

        attn_maps = model.get_attention_maps(image, aux)
        attn_14   = cls_attention_last_block(attn_maps)   # (B, 14, 14)

        with torch.no_grad():
            logits, _ = model(image, aux)

        for i in range(image.shape[0]):
            if collected >= max_samples:
                break
            a2d = attn_14[i].cpu().numpy()
            results["attention_2d"].append(a2d)
            results["profile_1d"].append(attention_to_lightcurve_profile(a2d))
            results["labels"].append(int(labels[i]))
            results["logits"].append(logits[i].cpu().numpy())
            collected += 1

    return results
