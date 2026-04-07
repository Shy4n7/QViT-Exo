"""ExoplanetViT and AuxMLP — ViT-B/16 backbone for exoplanet transit detection.

Architecture
-----------
- Backbone: ViT-B/16 from timm, modified for 2-channel input
- AuxMLP: maps auxiliary stellar/orbital features (B, aux_dim) → (B, 32)
- ExoplanetViT: backbone CLS token (B, 768) + AuxMLP output (B, 32)
    → class_logits (B, 2)
    → reg_preds    (B, 3) | None

Attention maps are extracted by registering forward hooks on each block's
attn_drop layer (fused attention is disabled at init time to ensure the
explicit attention matrix is materialised).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class AuxMLP(nn.Module):
    """Maps auxiliary features (B, aux_dim) → (B, out_dim=32).

    Parameters
    ----------
    aux_dim : int   Input dimension (default 5 stellar/orbital features).
    hidden  : int   Hidden layer size.
    out_dim : int   Output embedding dimension.
    """

    def __init__(
        self,
        aux_dim: int = 5,
        hidden: int = 64,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(aux_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, aux_dim) → (B, out_dim)
        return self._net(x)


class ExoplanetViT(nn.Module):
    """ViT-B/16 + AuxMLP for exoplanet transit classification.

    Parameters
    ----------
    pretrained       : bool  Load pretrained weights (default True).
    regression_head  : bool  Add a 3-output regression branch (default True).
    freeze_backbone  : bool  Freeze all backbone parameters (default False).
    aux_dim          : int   Auxiliary feature dimension (default 5).
    """

    _VIT_EMBED_DIM: int = 768   # ViT-B/16 CLS token dimension
    _AUX_OUT: int = 32           # AuxMLP output dimension
    _NUM_BLOCKS: int = 12        # ViT-B/16 transformer block count

    def __init__(
        self,
        pretrained: bool = True,
        regression_head: bool = True,
        freeze_backbone: bool = False,
        aux_dim: int = 5,
    ) -> None:
        super().__init__()

        # ViT-B/16 backbone — 2-channel input, no classifier head.
        # Returns CLS token of shape (B, 768).
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            in_chans=2,
            num_classes=0,
        )

        # Disable fused (flash) attention so the explicit attention matrix is
        # materialised, which allows forward hooks on attn_drop to capture it.
        for block in self.backbone.blocks:
            block.attn.fused_attn = False

        # Auxiliary feature branch
        self.aux_mlp = AuxMLP(aux_dim=aux_dim)

        # Combined feature dimension: CLS token (768) + aux embedding (32) = 800
        fc_in = self._VIT_EMBED_DIM + self._AUX_OUT

        # Optional regression head: predicts (period, radius, transit_depth).
        # When present, its output feeds INTO the classification head so that
        # a single classification backward() populates gradients for all parameters.
        self._uses_regression = regression_head
        if regression_head:
            self.reg_head = nn.Linear(fc_in, 3)
            # Classification head receives combined (800) + reg predictions (3) = 803
            self.class_head = nn.Linear(fc_in + 3, 2)
        else:
            # Classification head receives combined features only (800)
            self.class_head = nn.Linear(fc_in, 2)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        aux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the forward pass.

        Parameters
        ----------
        image : Tensor (B, 2, 224, 224)  2-channel phase-folded transit image.
        aux   : Tensor (B, aux_dim)       Auxiliary stellar/orbital features.

        Returns
        -------
        class_logits : Tensor (B, 2)         Raw classification logits.
        reg_preds    : Tensor (B, 3) | None  Physical parameter predictions.
        """
        cls_token = self.backbone(image)                              # (B, 768)
        aux_feat = self.aux_mlp(aux)                                   # (B, 32)
        combined = torch.cat([cls_token, aux_feat], dim=1)            # (B, 800)

        if self._uses_regression:
            reg_preds = self.reg_head(combined)                        # (B, 3)
            # Fuse physical parameter estimates into the classification path.
            # This couples both heads so a single classification backward()
            # produces non-None gradients for all trainable parameters.
            fused = torch.cat([combined, reg_preds], dim=1)           # (B, 803)
            class_logits = self.class_head(fused)                     # (B, 2)
        else:
            reg_preds = None
            class_logits = self.class_head(combined)                  # (B, 2)

        return class_logits, reg_preds

    # ------------------------------------------------------------------
    # Attention maps
    # ------------------------------------------------------------------

    def get_attention_maps(
        self,
        image: torch.Tensor,
        aux: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract per-block attention weights via forward hooks.

        Hooks are registered on each block's ``attn_drop`` layer.  Because fused
        attention is disabled at init time, ``attn_drop`` always receives the
        softmax-normalised attention matrix as its input.

        Returns
        -------
        dict[str, Tensor]
            Keys ``'block_0'`` through ``'block_11'``.
            Each tensor shape: ``(B, num_heads, seq_len, seq_len)``
            = ``(B, 12, 197, 197)`` for ViT-B/16 with 224×224 input.
        """
        attention_maps: dict[str, torch.Tensor] = {}
        hooks: list = []

        for i, block in enumerate(self.backbone.blocks):
            def _make_hook(idx: int):
                def _hook(
                    module: nn.Module,
                    inp: tuple[torch.Tensor, ...],
                    out: torch.Tensor,
                ) -> None:
                    # inp[0] is the softmaxed attention matrix before dropout
                    # shape: (B, num_heads, seq_len, seq_len)
                    attention_maps[f"block_{idx}"] = inp[0].detach()

                return _hook

            hooks.append(block.attn.attn_drop.register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            self.forward(image, aux)

        for hook in hooks:
            hook.remove()

        return attention_maps
