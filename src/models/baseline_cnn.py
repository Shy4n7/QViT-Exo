"""Shallue & Vanderburg 2018 baseline 1D CNN for exoplanet transit classification.

Reference: Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain
           around Kepler-80 and an Eighth Planet around Kepler-90
           https://arxiv.org/abs/1712.05044

Architecture summary
--------------------
Two optional input branches operating on 1D phase-folded flux:

  Global view  : 2001-point phase-folded flux covering the full orbital period.
  Local view   : 201-point zoom around the transit event (optional).

Each branch passes through 4 Conv1D blocks (Conv -> ReLU -> MaxPool), then the
feature vectors are concatenated (or kept as-is when local view is disabled),
fed through a 512-unit fully-connected layer with dropout, and finally a 2-unit
linear head that returns raw logits for binary classification.

Usage with nn.CrossEntropyLoss (no softmax in forward).
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Conv1d + ReLU + MaxPool1d.

    Parameters
    ----------
    in_channels:  Number of input channels.
    out_channels: Number of convolutional filters.
    kernel_size:  Convolution kernel width (default 5, matching S&V 2018).
    pool_size:    MaxPool kernel width.
    pool_stride:  MaxPool stride (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 5,
        pool_stride: int = 2,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same'-style padding to preserve length
        )
        self._relu = nn.ReLU()
        self._pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L) -> (B, C', L')
        return self._pool(self._relu(self._conv(x)))


def _build_conv_branch() -> nn.Sequential:
    """Build the 4-block convolutional feature extractor shared by both views.

    Block 1-2 use pool_size=5; blocks 3-4 use pool_size=7 (S&V 2018 §2.2).
    Channel progression: 1 -> 16 -> 32 -> 64 -> 128
    """
    return nn.Sequential(
        ConvBlock1D(1,   16,  kernel_size=5, pool_size=5, pool_stride=2),
        ConvBlock1D(16,  32,  kernel_size=5, pool_size=5, pool_stride=2),
        ConvBlock1D(32,  64,  kernel_size=5, pool_size=7, pool_stride=2),
        ConvBlock1D(64,  128, kernel_size=5, pool_size=7, pool_stride=2),
    )


class BaselineCNN(nn.Module):
    """1D CNN reproducing the Shallue & Vanderburg 2018 architecture.

    After the 4 convolutional blocks each branch is reduced to a fixed-size
    128-dimensional vector via Global Average Pooling (GAP).  GAP collapses
    the variable-length temporal axis to a single value per channel, making
    the fully-connected head independent of input length and keeping the
    total parameter count well under 500K.

    Parameters
    ----------
    use_local_view : bool
        When True a second 201-point local-view branch is added and its
        128-d GAP vector is concatenated with the global branch's 128-d
        vector before the FC head.  Default: False.
    dropout : float
        Dropout probability applied after the first FC layer.  Default: 0.5.
    """

    _FC_HIDDEN: int = 512
    # After GAP each branch produces exactly 128 features (one per channel).
    _BRANCH_FEATURES: int = 128

    def __init__(
        self,
        use_local_view: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self._use_local_view = use_local_view

        # --- Convolutional branches ---
        self._global_branch = _build_conv_branch()

        if use_local_view:
            self._local_branch = _build_conv_branch()

        # --- FC input dimension ---
        # Global branch contributes 128 features; local branch adds another 128.
        fc_in = self._BRANCH_FEATURES * (2 if use_local_view else 1)

        # --- Fully-connected head ---
        self._fc = nn.Sequential(
            nn.Linear(fc_in, self._FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self._FC_HIDDEN, 2),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        global_view : Tensor of shape (B, 1, 2001)
            Phase-folded flux covering the full orbital period.
        local_view : Tensor of shape (B, 1, 201) or None
            Zoomed transit view.  Required when use_local_view=True.

        Returns
        -------
        Tensor of shape (B, 2) — raw logits (no softmax).
        """
        global_features = self._global_branch(global_view)       # (B, 128, L_g)
        global_pooled = global_features.mean(dim=2)               # (B, 128) via GAP

        if self._use_local_view:
            if local_view is None:
                raise ValueError(
                    "local_view tensor is required when use_local_view=True"
                )
            local_features = self._local_branch(local_view)       # (B, 128, L_l)
            local_pooled = local_features.mean(dim=2)             # (B, 128) via GAP
            combined = torch.cat([global_pooled, local_pooled], dim=1)  # (B, 256)
        else:
            combined = global_pooled                              # (B, 128)

        return self._fc(combined)  # (B, 2) raw logits
