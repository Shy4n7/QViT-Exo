"""Tests for ExoplanetQuantumViT — Phase 3 quantum hybrid model.

Covers both quantum_mode variants:
  - 'vqc_head'   : VQC classification head (Unlu 2024 approach)
  - 'qonn_attn'  : QONN-augmented CLS-token refinement (Tesi 2024 approach)

All tests use a small synthetic batch (B=2, 224×224) and tiny quantum
circuits (n_qubits=2, n_layers=1) to keep CPU wall-time acceptable.

TDD contract:
    - Tests were written to describe the expected interface.
    - Implementation in src/models/quantum_vit.py must satisfy all assertions.
    - Tests must NOT be modified to pass — only the implementation changes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE     = 2
AUX_DIM        = 5
IMG_CHANNELS   = 2
IMG_SIZE       = 224
N_QUBITS       = 2   # minimal for fast CPU tests
N_LAYERS       = 1
VIT_NUM_BLOCKS = 12
VIT_SEQ_LEN    = 197   # 14×14 patches + 1 CLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Small (B=2) image + aux batch."""
    image = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    aux   = torch.randn(BATCH_SIZE, AUX_DIM)
    return image, aux


@pytest.fixture(scope="module")
def vqc_model() -> "ExoplanetQuantumViT":
    from src.models.quantum_vit import ExoplanetQuantumViT
    return ExoplanetQuantumViT(
        quantum_mode="vqc_head",
        n_qubits=N_QUBITS,
        n_quantum_layers=N_LAYERS,
        pretrained=False,
        regression_head=True,
    )


@pytest.fixture(scope="module")
def qonn_model() -> "ExoplanetQuantumViT":
    from src.models.quantum_vit import ExoplanetQuantumViT
    return ExoplanetQuantumViT(
        quantum_mode="qonn_attn",
        n_qubits=N_QUBITS,
        n_quantum_layers=N_LAYERS,
        pretrained=False,
        regression_head=True,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_vqc_head_constructs(self) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        model = ExoplanetQuantumViT(
            quantum_mode="vqc_head",
            n_qubits=N_QUBITS,
            n_quantum_layers=N_LAYERS,
            pretrained=False,
        )
        assert isinstance(model, nn.Module)

    def test_qonn_attn_constructs(self) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        model = ExoplanetQuantumViT(
            quantum_mode="qonn_attn",
            n_qubits=N_QUBITS,
            n_quantum_layers=N_LAYERS,
            pretrained=False,
        )
        assert isinstance(model, nn.Module)

    def test_invalid_quantum_mode_raises(self) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        with pytest.raises(ValueError, match="quantum_mode"):
            ExoplanetQuantumViT(quantum_mode="invalid", pretrained=False)  # type: ignore[arg-type]

    def test_vqc_model_has_quantum_parameters(self, vqc_model) -> None:
        assert len(vqc_model.quantum_parameters) > 0

    def test_qonn_model_has_quantum_parameters(self, qonn_model) -> None:
        assert len(qonn_model.quantum_parameters) > 0

    def test_quantum_params_require_grad(self, vqc_model) -> None:
        for p in vqc_model.quantum_parameters:
            assert p.requires_grad

    def test_backbone_frozen_when_requested(self) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        model = ExoplanetQuantumViT(
            quantum_mode="vqc_head",
            n_qubits=N_QUBITS,
            n_quantum_layers=N_LAYERS,
            pretrained=False,
            freeze_backbone=True,
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad


# ---------------------------------------------------------------------------
# Forward pass — output shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_vqc_class_logits_shape(
        self, vqc_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        logits, _ = vqc_model(image, aux)
        assert logits.shape == (BATCH_SIZE, 2), f"Expected (B,2), got {logits.shape}"

    def test_vqc_reg_preds_shape(
        self, vqc_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        _, reg = vqc_model(image, aux)
        assert reg is not None
        assert reg.shape == (BATCH_SIZE, 3), f"Expected (B,3), got {reg.shape}"

    def test_qonn_class_logits_shape(
        self, qonn_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        logits, _ = qonn_model(image, aux)
        assert logits.shape == (BATCH_SIZE, 2)

    def test_qonn_reg_preds_shape(
        self, qonn_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        _, reg = qonn_model(image, aux)
        assert reg is not None
        assert reg.shape == (BATCH_SIZE, 3)

    def test_no_regression_returns_none(self, dummy_batch) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        model = ExoplanetQuantumViT(
            quantum_mode="vqc_head",
            n_qubits=N_QUBITS,
            n_quantum_layers=N_LAYERS,
            pretrained=False,
            regression_head=False,
        )
        image, aux = dummy_batch
        _, reg = model(image, aux)
        assert reg is None

    def test_output_dtype_float32(self, vqc_model, dummy_batch) -> None:
        image, aux = dummy_batch
        logits, _ = vqc_model(image, aux)
        assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through_vqc_circuit(
        self, vqc_model, dummy_batch
    ) -> None:
        """Backward pass must populate gradients on quantum parameters."""
        image, aux = dummy_batch
        logits, _ = vqc_model(image, aux)
        loss = logits.sum()
        loss.backward()

        q_params = vqc_model.quantum_parameters
        assert any(p.grad is not None for p in q_params), \
            "No gradients on quantum parameters after backward()"
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q_params), \
            "All quantum parameter gradients are zero"

    def test_gradients_flow_through_qonn_circuit(
        self, qonn_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        logits, _ = qonn_model(image, aux)
        loss = logits.sum()
        loss.backward()

        q_params = qonn_model.quantum_parameters
        assert any(p.grad is not None for p in q_params)

    def test_backbone_grad_none_when_frozen(self, dummy_batch) -> None:
        from src.models.quantum_vit import ExoplanetQuantumViT
        model = ExoplanetQuantumViT(
            quantum_mode="vqc_head",
            n_qubits=N_QUBITS,
            n_quantum_layers=N_LAYERS,
            pretrained=False,
            freeze_backbone=True,
        )
        image, aux = dummy_batch
        logits, _ = model(image, aux)
        logits.sum().backward()
        for param in model.backbone.parameters():
            assert param.grad is None


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------

class TestAttentionMaps:
    def test_attention_maps_return_dict(
        self, vqc_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        maps = vqc_model.get_attention_maps(image, aux)
        assert isinstance(maps, dict)

    def test_attention_maps_all_blocks(
        self, vqc_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        maps = vqc_model.get_attention_maps(image, aux)
        assert len(maps) == VIT_NUM_BLOCKS
        for i in range(VIT_NUM_BLOCKS):
            assert f"block_{i}" in maps

    def test_attention_map_shape(
        self, vqc_model, dummy_batch
    ) -> None:
        image, aux = dummy_batch
        maps = vqc_model.get_attention_maps(image, aux)
        attn = maps["block_0"]
        # (B, num_heads, seq_len, seq_len) = (2, 12, 197, 197)
        assert attn.shape[0] == BATCH_SIZE
        assert attn.shape[2] == VIT_SEQ_LEN
        assert attn.shape[3] == VIT_SEQ_LEN

    def test_attention_maps_do_not_modify_model_state(
        self, vqc_model, dummy_batch
    ) -> None:
        """get_attention_maps must leave model in eval-compatible state."""
        image, aux = dummy_batch
        vqc_model.eval()
        vqc_model.get_attention_maps(image, aux)
        # Model should still produce outputs after hook extraction
        logits, _ = vqc_model(image, aux)
        assert logits.shape == (BATCH_SIZE, 2)


# ---------------------------------------------------------------------------
# BarrenPlateauMonitor
# ---------------------------------------------------------------------------

class TestBarrenPlateauMonitor:
    def test_monitor_returns_float(self, vqc_model, dummy_batch) -> None:
        from scripts.train_quantum_vit import BarrenPlateauMonitor
        monitor = BarrenPlateauMonitor(vqc_model.quantum_parameters)
        image, aux = dummy_batch
        logits, _ = vqc_model(image, aux)
        logits.sum().backward()
        variance = monitor.step()
        assert isinstance(variance, float)

    def test_monitor_history_grows(self, vqc_model, dummy_batch) -> None:
        from scripts.train_quantum_vit import BarrenPlateauMonitor
        monitor = BarrenPlateauMonitor(vqc_model.quantum_parameters)
        for _ in range(3):
            image, aux = dummy_batch
            vqc_model.zero_grad()
            logits, _ = vqc_model(image, aux)
            logits.sum().backward()
            monitor.step()
        assert len(monitor.history) == 3

    def test_monitor_zero_when_no_grads(self, vqc_model) -> None:
        from scripts.train_quantum_vit import BarrenPlateauMonitor
        vqc_model.zero_grad()
        monitor = BarrenPlateauMonitor(vqc_model.quantum_parameters)
        variance = monitor.step()
        assert variance == 0.0
