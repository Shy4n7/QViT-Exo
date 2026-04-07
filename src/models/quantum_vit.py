"""Phase 3 — Hybrid Quantum-Classical ViT for exoplanet transit vetting.

Two quantum integration modes:

  ``vqc_head``   VQC replaces the linear classification head (Unlu 2024).
                 Angle-embed the compressed CLS token into n_qubits, apply
                 StronglyEntanglingLayers, measure in Z-basis, classify.
                 Simpler integration; lower barren-plateau risk at small n_qubits.

  ``qonn_attn``  QONN-augmented CLS-token refinement (Tesi 2024 inspired).
                 Compress CLS token to n_qubits → BasicEntanglerLayers (orthogonal-
                 like) → expand back to 768 → residual-add to CLS token.
                 More novel; natural fit for attention's orthogonal structure.

Both variants:
  - Share the ExoplanetViT backbone (timm ViT-B/16) and AuxMLP unchanged
  - Accept identical forward(image, aux) → (class_logits, reg_preds) signature
    so ViTTrainer can be reused without modification
  - Expose get_attention_maps() for interpretability

Forward data flow
-----------------
  vqc_head:
      backbone → CLS(768)
      AuxMLP(5) → aux_feat(32)
      combined = cat[CLS, aux_feat] (800)
      down(800→n_q) → AngleEmbed → StronglyEntangling → ⟨Z⟩(n_q)
      up(n_q→2) → class_logits

  qonn_attn:
      backbone → CLS(768)
      down(768→n_q) → AngleEmbed → BasicEntangler → ⟨Z⟩(n_q) → up(n_q→768)
      cls_refined = CLS + quantum_residual (residual keeps gradient flowing)
      AuxMLP(5) → aux_feat(32)
      combined = cat[cls_refined, aux_feat] (800)
      class_head(800→2) → class_logits
"""

from __future__ import annotations

import math
from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

from src.models.vit_model import AuxMLP, ExoplanetViT

# ---------------------------------------------------------------------------
# Quantum circuit factories
# ---------------------------------------------------------------------------

def _make_vqc_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Return a PennyLane TorchLayer implementing a variational quantum circuit.

    Architecture:
        AngleEmbedding (Y-rotation) → StronglyEntanglingLayers → ⟨Z_i⟩

    StronglyEntanglingLayers provide maximum expressivity for classification
    tasks at moderate qubit counts (4–8 qubits).

    Parameters
    ----------
    n_qubits : int  Number of qubits (= input/output dimension).
    n_layers : int  Number of variational layers.

    Returns
    -------
    qml.qnn.TorchLayer  Differentiable via backprop through the state vector.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(_circuit, weight_shapes)


def _make_qonn_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """Return a QONN-like PennyLane TorchLayer.

    Architecture:
        AngleEmbedding (Y-rotation) → BasicEntanglerLayers (RY + CNOT ring)

    BasicEntanglerLayers implement RY rotations followed by nearest-neighbour
    CNOT chains — a simplified quantum orthogonal network (QONN) approximation.
    Fewer parameters than StronglyEntangling → better gradient flow → lower
    barren-plateau risk.  Natural for orthogonal attention operations (Tesi 2024).

    Parameters
    ----------
    n_qubits : int  Number of qubits.
    n_layers : int  Number of layers.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(_circuit, weight_shapes)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _VQCHead(nn.Module):
    """VQC classification head (Unlu 2024 approach).

    combined_dim → down_proj(n_qubits) → VQC(n_qubits) → up_proj(2)
    """

    def __init__(self, combined_dim: int, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.down = nn.Linear(combined_dim, n_qubits)
        self.qlayer = _make_vqc_layer(n_qubits, n_layers)
        self.up = nn.Linear(n_qubits, 2)
        self._n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, combined_dim)
        z = torch.tanh(self.down(x))  # tanh keeps values in [-1,1] for AngleEmbed
        z = z * math.pi               # scale to [-π, π] for full rotation range
        q_out = self.qlayer(z)        # (B, n_qubits) — expectation values in [-1,1]
        return self.up(q_out)         # (B, 2)


class _QONNRefinement(nn.Module):
    """QONN-augmented CLS-token refinement (Tesi 2024 inspired).

    CLS(768) → down(n_qubits) → QONN → up(768) → residual + CLS
    """

    def __init__(self, embed_dim: int, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, n_qubits)
        self.qlayer = _make_qonn_layer(n_qubits, n_layers)
        self.up = nn.Linear(n_qubits, embed_dim)
        self._scale = nn.Parameter(torch.tensor(0.1))  # learnable residual scale

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        # cls_token: (B, 768)
        z = torch.tanh(self.down(cls_token)) * math.pi   # (B, n_qubits)
        q_out = self.qlayer(z)                            # (B, n_qubits)
        refined = self.up(q_out)                          # (B, 768)
        return cls_token + self._scale * refined          # residual (B, 768)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

QuantumMode = Literal["vqc_head", "qonn_attn"]

_VIT_EMBED_DIM = 768
_AUX_OUT = 32
_FC_IN = _VIT_EMBED_DIM + _AUX_OUT   # 800


class ExoplanetQuantumViT(nn.Module):
    """Hybrid Quantum-Classical ViT for exoplanet transit classification.

    Accepts the same ``forward(image, aux)`` signature as the classical
    ``ExoplanetViT`` so existing ``ViTTrainer`` infrastructure is reused.

    Parameters
    ----------
    quantum_mode     : 'vqc_head' | 'qonn_attn'
    n_qubits         : Number of qubits in the quantum circuit (default 4).
    n_quantum_layers : Depth of the variational ansatz (default 2).
    pretrained       : Load pretrained ViT-B/16 backbone weights.
    regression_head  : Add a 3-output transit parameter regression branch.
    freeze_backbone  : Freeze all backbone parameters (train quantum head only).
    aux_dim          : Number of auxiliary stellar/orbital features.
    """

    def __init__(
        self,
        quantum_mode:        QuantumMode = "vqc_head",
        n_qubits:            int  = 4,
        n_quantum_layers:    int  = 2,
        pretrained:          bool = True,
        regression_head:     bool = True,
        freeze_backbone:     bool = False,
        aux_dim:             int  = 5,
    ) -> None:
        super().__init__()

        self._quantum_mode = quantum_mode
        self._uses_regression = regression_head

        # ── Backbone + AuxMLP (shared with classical model) ──────
        # Re-use the classical ExoplanetViT as a feature extractor.
        # We do NOT use its heads — only backbone + aux_mlp.
        _base = ExoplanetViT(
            pretrained=pretrained,
            regression_head=False,   # we add our own heads below
            freeze_backbone=freeze_backbone,
            aux_dim=aux_dim,
        )
        self.backbone = _base.backbone
        self.aux_mlp  = _base.aux_mlp

        # Disable fused attention so attention hooks work in eval
        for block in self.backbone.blocks:
            block.attn.fused_attn = False

        # ── Quantum components ────────────────────────────────────
        if quantum_mode == "vqc_head":
            # Regression head operates on classical combined features (800)
            if regression_head:
                self.reg_head   = nn.Linear(_FC_IN, 3)
                self.vqc_head   = _VQCHead(_FC_IN + 3, n_qubits, n_quantum_layers)
            else:
                self.vqc_head   = _VQCHead(_FC_IN, n_qubits, n_quantum_layers)

        elif quantum_mode == "qonn_attn":
            self.qonn = _QONNRefinement(_VIT_EMBED_DIM, n_qubits, n_quantum_layers)
            # After QONN residual the combined_dim is still 800
            if regression_head:
                self.reg_head   = nn.Linear(_FC_IN, 3)
                self.class_head = nn.Linear(_FC_IN + 3, 2)
            else:
                self.class_head = nn.Linear(_FC_IN, 2)

        else:
            raise ValueError(
                f"quantum_mode must be 'vqc_head' or 'qonn_attn', got {quantum_mode!r}"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        aux:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the hybrid forward pass.

        Parameters
        ----------
        image : Tensor (B, 2, 224, 224)
        aux   : Tensor (B, aux_dim)

        Returns
        -------
        class_logits : Tensor (B, 2)
        reg_preds    : Tensor (B, 3) | None
        """
        cls_token = self.backbone(image)    # (B, 768)
        aux_feat  = self.aux_mlp(aux)       # (B, 32)

        if self._quantum_mode == "vqc_head":
            combined  = torch.cat([cls_token, aux_feat], dim=1)   # (B, 800)
            reg_preds = None
            if self._uses_regression:
                reg_preds = self.reg_head(combined)                # (B, 3)
                fused     = torch.cat([combined, reg_preds], dim=1) # (B, 803)
            else:
                fused = combined
            class_logits = self.vqc_head(fused)                    # (B, 2)

        else:  # qonn_attn
            cls_refined  = self.qonn(cls_token)                    # (B, 768)
            combined     = torch.cat([cls_refined, aux_feat], dim=1) # (B, 800)
            reg_preds    = None
            if self._uses_regression:
                reg_preds    = self.reg_head(combined)             # (B, 3)
                fused        = torch.cat([combined, reg_preds], dim=1) # (B, 803)
            else:
                fused = combined
            class_logits = self.class_head(fused)                  # (B, 2)

        return class_logits, reg_preds

    # ------------------------------------------------------------------
    # Attention maps (re-use classical hook mechanism on shared backbone)
    # ------------------------------------------------------------------

    def get_attention_maps(
        self,
        image: torch.Tensor,
        aux:   torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract per-block attention weights — same API as ExoplanetViT."""
        attention_maps: dict[str, torch.Tensor] = {}
        hooks: list = []

        for i, block in enumerate(self.backbone.blocks):
            def _make_hook(idx: int):
                def _hook(module, inp, out):
                    attention_maps[f"block_{idx}"] = inp[0].detach()
                return _hook
            hooks.append(block.attn.attn_drop.register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            self.forward(image, aux)

        for hook in hooks:
            hook.remove()

        return attention_maps

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def quantum_parameters(self) -> list[nn.Parameter]:
        """Parameters belonging to the quantum circuit layer(s) only."""
        if self._quantum_mode == "vqc_head":
            return list(self.vqc_head.qlayer.parameters())
        return list(self.qonn.qlayer.parameters())
