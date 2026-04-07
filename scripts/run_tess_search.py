"""Phase 6 — Apply trained model to unvetted TESS TOI candidates.

Pipeline
--------
1. Fetch/load TESS TOI catalog (ExoFOP).
2. Filter to unvetted Planet Candidates (PC/APC) with valid period + epoch.
3. Cross-match against NASA confirmed planets; remove already-known objects.
4. For each remaining TOI (up to --max-candidates):
   a. Download TESS PDCSAP light curve via lightkurve.
   b. Preprocess: detrend → normalise → sigma-clip → phase-fold.
   c. Generate 2-channel image (Recurrence Plot + GADF, 64×64).
   d. Extract auxiliary diagnostic features.
   e. Run model inference → planet probability + attention maps.
   f. Compute attention coherence score (fraction of attention on ingress/egress).
5. Filter by confidence threshold AND coherence threshold.
6. Output ranked CSV + plain-text summary.

Outputs
-------
results/tess_search/
    candidates.csv          — ranked list of high-confidence new candidates
    all_scored.csv          — every processed TOI with all scores
    search_report.txt       — human-readable summary

Usage
-----
python scripts/run_tess_search.py \\
    --checkpoint models/quantum_vit/best_model.pt \\
    --model-type quantum \\
    --quantum-mode qonn_attn \\
    --toi-catalog data/toi_catalog.csv \\
    --cache-dir data/tess_cache \\
    --sectors 70 71 72 73 74 75 76 77 78 79 80 81 82 83 \\
    --confidence-threshold 0.90 \\
    --coherence-threshold 0.20 \\
    --max-candidates 500
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, Resize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.tess_download import download_tess_lightcurve
from src.data.tess_catalog import (
    fetch_toi_catalog,
    fetch_confirmed_tic_ids,
    filter_unvetted,
    remove_confirmed,
    extract_toi_params,
)
from src.data.preprocess import preprocess_pipeline
from src.data.imaging import generate_image_pair
from src.data.auxiliary import extract_auxiliary
from src.models.vit_model import ExoplanetViT
from src.models.quantum_vit import ExoplanetQuantumViT
from src.interpretability.attention_analysis import (
    cls_attention_last_block,
    attention_to_lightcurve_profile,
    ingress_egress_indicator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Image resize to ViT-B/16 input size
_RESIZE_224 = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)

# NaN sentinels matching dataset.py
_NAN_SENTINELS: dict[int, float] = {2: 1.0, 4: -1.0}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_quantum(path: str, mode: str) -> ExoplanetQuantumViT:
    m = ExoplanetQuantumViT(quantum_mode=mode, pretrained=False, regression_head=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def _load_classical(path: str) -> ExoplanetViT:
    m = ExoplanetViT(pretrained=False, regression_head=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _aux_to_tensor(aux_arr: np.ndarray) -> torch.Tensor:
    """Apply NaN sentinels and convert to float32 tensor."""
    arr = aux_arr.astype(np.float32).copy()
    for idx, sentinel in _NAN_SENTINELS.items():
        if idx < len(arr) and np.isnan(arr[idx]):
            arr[idx] = np.float32(sentinel)
    arr = np.where(np.isnan(arr), np.float32(0.0), arr)
    return torch.from_numpy(arr)


def _process_lightcurve(
    lc: object,
    period_days: float,
    epoch_btjd: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Preprocess a lightkurve LightCurve into (image_tensor, aux_tensor).

    Returns None when preprocessing fails (insufficient data, etc.).
    """
    try:
        time_arr  = np.array(lc.time.value,   dtype=np.float64)
        flux_arr  = np.array(lc.flux.value,   dtype=np.float64)
    except Exception as exc:
        logger.debug("LC attribute access failed: %s", exc)
        return None

    if len(flux_arr) < 50:
        logger.debug("Too few cadences (%d) after download.", len(flux_arr))
        return None

    # Full preprocessing pipeline: detrend → normalise → sigma-clip → phase-fold
    try:
        plc = preprocess_pipeline(time_arr, flux_arr, period=period_days, epoch=epoch_btjd)
    except Exception as exc:
        logger.debug("preprocess_pipeline failed: %s", exc)
        return None

    # 2-channel image (RP + GADF, 64×64)
    try:
        image_np = generate_image_pair(plc.flux, size=64).astype(np.float32)
    except Exception as exc:
        logger.debug("generate_image_pair failed: %s", exc)
        return None

    image_t = torch.from_numpy(image_np)
    image_t = _RESIZE_224(image_t)  # (2, 224, 224)

    # Auxiliary diagnostics
    try:
        aux = extract_auxiliary(
            flux=flux_arr,
            time=time_arr,
            phase=plc.phase,
            period=period_days,
            epoch=epoch_btjd,
        )
        aux_np = np.array([
            aux.odd_depth,
            aux.even_depth,
            aux.depth_ratio,
            float(aux.secondary_eclipse),
            aux.centroid_shift,
        ], dtype=np.float32)
    except Exception as exc:
        logger.debug("extract_auxiliary failed: %s", exc)
        aux_np = np.zeros(5, dtype=np.float32)

    aux_t = _aux_to_tensor(aux_np)
    return image_t, aux_t


# ---------------------------------------------------------------------------
# Attention coherence score
# ---------------------------------------------------------------------------

def _attention_coherence(
    profile: np.ndarray,
    duration_fraction: float,
    t_mid: float = 0.5,
) -> float:
    """Fraction of attention concentrated on ingress + egress windows.

    High coherence (→ 1.0) means the model attends to the physically meaningful
    transit shoulder regions.  Low coherence (→ 0.0) means attention is diffuse
    or focused elsewhere (EB or noise diagnostic).

    Parameters
    ----------
    profile : ndarray (N,)
        1-D normalised attention profile from ``attention_to_lightcurve_profile``.
    duration_fraction : float
        Transit duration as a fraction of the phase window (= duration_hours /
        (period_days × 24)).  Clamped to [0.01, 0.45].
    t_mid : float
        Transit centre in normalised [0, 1] coordinates.  Always 0.5 for
        phase-folded light curves.

    Returns
    -------
    float
        Coherence score in [0, 1].
    """
    duration = float(np.clip(duration_fraction, 0.01, 0.45))
    n = len(profile)
    indicator = ingress_egress_indicator(n, t_mid=t_mid, duration=duration)
    total = float(profile.sum())
    if total < 1e-10:
        return 0.0
    return float((profile * indicator).sum() / total)


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: torch.nn.Module,
    image_t: torch.Tensor,
    aux_t: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Run model inference on a single sample.

    Returns
    -------
    dict with keys:
        planet_prob     float — softmax probability for class 1 (planet)
        coherence       float — attention coherence score
        attn_entropy    float — entropy of the CLS attention distribution
    """
    image_b = image_t.unsqueeze(0).to(device)
    aux_b   = aux_t.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(image_b, aux_b)
        probs = F.softmax(logits, dim=-1)
        planet_prob = float(probs[0, 1].item())

        attn_maps = model.get_attention_maps(image_b, aux_b)

    attn_14   = cls_attention_last_block(attn_maps)    # (1, 14, 14)
    attn_2d   = attn_14[0].cpu().numpy()               # (14, 14)
    profile   = attention_to_lightcurve_profile(attn_2d)  # (14,)

    # Entropy of attention distribution (higher = more diffuse)
    p_flat  = attn_2d.flatten()
    p_norm  = p_flat / (p_flat.sum() + 1e-12)
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

    return {
        "planet_prob":  planet_prob,
        "profile":      profile,
        "attn_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------

def _build_report(
    candidates_df: pd.DataFrame,
    all_df: pd.DataFrame,
    conf_thresh: float,
    coherence_thresh: float,
) -> str:
    lines = [
        "=" * 60,
        "  PHASE 6 — TESS SEARCH RESULTS",
        "=" * 60,
        f"  Total TOIs processed       : {len(all_df)}",
        f"  Confidence threshold       : {conf_thresh:.2f}",
        f"  Coherence threshold        : {coherence_thresh:.2f}",
        f"  Candidates above threshold : {len(candidates_df)}",
        "",
        "  Top 10 candidates by planet probability",
        "  " + "-" * 40,
    ]
    top10 = candidates_df.head(10)
    for _, row in top10.iterrows():
        lines.append(
            f"  TOI {row['toi_id']:<10.2f}  TIC {int(row['tic_id']):<12d}  "
            f"prob={row['planet_prob']:.4f}  coherence={row['coherence']:.4f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 — TESS unvetted TOI screening")
    p.add_argument("--checkpoint",            required=True,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--model-type",            default="quantum",
                   choices=["quantum", "classical"])
    p.add_argument("--quantum-mode",          default="qonn_attn",
                   choices=["vqc_head", "qonn_attn"])
    p.add_argument("--toi-catalog",           default=None,
                   help="Path to pre-downloaded TOI CSV. If absent, fetches fresh.")
    p.add_argument("--toi-catalog-output",    default="data/toi_catalog.csv")
    p.add_argument("--cache-dir",             default="data/tess_cache")
    p.add_argument("--sectors",               nargs="*", type=int, default=None,
                   help="TESS sector numbers to restrict download (e.g. 70 71 72).")
    p.add_argument("--exptime",               default="short",
                   choices=["short", "long", "fast"])
    p.add_argument("--confidence-threshold",  type=float, default=0.90)
    p.add_argument("--coherence-threshold",   type=float, default=0.15)
    p.add_argument("--max-candidates",        type=int,   default=500)
    p.add_argument("--results-dir",           default="results/tess_search/")
    p.add_argument("--skip-crossmatch",       action="store_true",
                   help="Skip NASA confirmed-planet cross-match (faster, offline mode).")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res    = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    logger.info("Loading %s model from %s…", args.model_type, args.checkpoint)
    if args.model_type == "quantum":
        model = _load_quantum(args.checkpoint, args.quantum_mode)
    else:
        model = _load_classical(args.checkpoint)
    model.to(device)
    logger.info("Model ready on %s.", device)

    # --- Fetch / load TOI catalog ---
    if args.toi_catalog and Path(args.toi_catalog).exists():
        logger.info("Loading TOI catalog from %s…", args.toi_catalog)
        toi_df = pd.read_csv(args.toi_catalog)
        toi_df.columns = [c.strip() for c in toi_df.columns]
    else:
        logger.info("Downloading fresh TOI catalog…")
        toi_df = fetch_toi_catalog(args.toi_catalog_output)

    logger.info("Raw catalog: %d rows.", len(toi_df))

    # --- Filter to unvetted candidates ---
    unvetted_df = filter_unvetted(toi_df)
    logger.info("Unvetted (PC/APC) with valid params: %d TOIs.", len(unvetted_df))

    # --- Cross-match against confirmed planets ---
    if not args.skip_crossmatch:
        confirmed_ids = fetch_confirmed_tic_ids()
        unvetted_df = remove_confirmed(unvetted_df, confirmed_ids)
        logger.info("After removing confirmed: %d TOIs remain.", len(unvetted_df))
    else:
        logger.info("Skipping confirmed-planet cross-match (--skip-crossmatch).")

    # Limit to --max-candidates
    to_process = unvetted_df.head(args.max_candidates)
    logger.info("Processing %d TOI candidates…", len(to_process))

    # --- Inference loop ---
    rows: list[dict] = []
    n_ok = 0
    n_fail = 0

    for idx, row in to_process.iterrows():
        try:
            params = extract_toi_params(row)
        except Exception as exc:
            logger.debug("extract_toi_params failed for row %d: %s", idx, exc)
            n_fail += 1
            continue

        tic_id       = params["tic_id"]
        toi_id       = params["toi_id"]
        period_days  = params["period_days"]
        epoch_btjd   = params["epoch_btjd"]
        duration_hrs = params["duration_hours"]

        logger.info(
            "[%d/%d] TOI %s  TIC %d  P=%.4fd  T0=%.4f  dur=%.2fh",
            n_ok + n_fail + 1, len(to_process),
            toi_id, tic_id, period_days, epoch_btjd, duration_hrs,
        )

        # Download light curve
        lc = download_tess_lightcurve(
            tic_id=tic_id,
            cache_dir=args.cache_dir,
            sectors=args.sectors,
            exptime=args.exptime,
        )
        if lc is None:
            logger.debug("TIC %d: light curve unavailable, skipping.", tic_id)
            n_fail += 1
            continue

        # Preprocess → image + aux tensors
        result = _process_lightcurve(lc, period_days, epoch_btjd)
        if result is None:
            logger.debug("TIC %d: preprocessing failed, skipping.", tic_id)
            n_fail += 1
            continue

        image_t, aux_t = result

        # Inference
        try:
            inf = _run_inference(model, image_t, aux_t, device)
        except Exception as exc:
            logger.warning("TIC %d: inference failed: %s", tic_id, exc)
            logger.debug(traceback.format_exc())
            n_fail += 1
            continue

        duration_fraction = duration_hrs / (period_days * 24.0)
        coherence = _attention_coherence(inf["profile"], duration_fraction)

        rows.append({
            "toi_id":        toi_id,
            "tic_id":        tic_id,
            "period_days":   period_days,
            "epoch_btjd":    epoch_btjd,
            "duration_hours": duration_hrs,
            "planet_prob":   inf["planet_prob"],
            "coherence":     coherence,
            "attn_entropy":  inf["attn_entropy"],
        })
        n_ok += 1

    logger.info("Inference complete: %d succeeded, %d skipped/failed.", n_ok, n_fail)

    if not rows:
        logger.warning("No TOIs were successfully processed. Exiting without output.")
        return

    all_df = (
        pd.DataFrame(rows)
        .sort_values("planet_prob", ascending=False)
        .reset_index(drop=True)
    )
    all_df.to_csv(res / "all_scored.csv", index=False)
    logger.info("All scored TOIs saved → %s", res / "all_scored.csv")

    # --- Filter to high-confidence candidates ---
    candidates_df = all_df.loc[
        (all_df["planet_prob"] >= args.confidence_threshold)
        & (all_df["coherence"] >= args.coherence_threshold)
    ].copy().reset_index(drop=True)

    candidates_df.to_csv(res / "candidates.csv", index=False)
    logger.info(
        "High-confidence candidates: %d  →  %s",
        len(candidates_df), res / "candidates.csv",
    )

    # --- Write report ---
    report = _build_report(
        candidates_df, all_df,
        args.confidence_threshold, args.coherence_threshold,
    )
    logger.info("\n%s", report)
    (res / "search_report.txt").write_text(report)
    logger.info("Search report saved → %s", res / "search_report.txt")

    logger.info("Phase 6 complete.")


if __name__ == "__main__":
    main()
