"""Tests for ExoplanetViT and AuxMLP — written BEFORE implementation (TDD Red phase).

TDD intent
----------
Every test in this file is intentionally written against a module
(src/models/vit_model.py) that does not exist yet.  Running the suite
now should produce import errors or AttributeErrors — that is the
expected RED state.

Once vit_model.py is implemented the suite must reach GREEN (all pass)
without modifying a single assertion here.  Only then is refactoring
permitted, with the constraint that the suite stays green throughout.

Architecture reference
----------------------
- ViT-B/16 backbone via timm (2-channel input projection)
- AuxMLP: (B, aux_dim) → (B, aux_out)
- ExoplanetViT: image (B, 2, 224, 224) + aux (B, 5)
    → class_logits (B, 2)
    → reg_preds   (B, 3) | None
- get_attention_maps: returns per-block attention weights
  key 'block_N', shape (B, num_heads, num_patches+1, num_patches+1)
  For ViT-B/16 with 224×224 input: (B, 12, 197, 197)
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 2
AUX_DIM: int = 5
IMG_H: int = 224
IMG_W: int = 224
IMG_CHANNELS: int = 2

# ViT-B/16 architecture constants
VIT_NUM_HEADS: int = 12
VIT_NUM_BLOCKS: int = 12
VIT_PATCH_SIZE: int = 16
VIT_NUM_PATCHES: int = (IMG_H // VIT_PATCH_SIZE) * (IMG_W // VIT_PATCH_SIZE)  # 196
VIT_SEQ_LEN: int = VIT_NUM_PATCHES + 1  # +1 for [CLS] token → 197


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_batch() -> torch.Tensor:
    """Random 2-channel image batch, shape (BATCH_SIZE, 2, 224, 224)."""
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_H, IMG_W)


@pytest.fixture
def aux_batch() -> torch.Tensor:
    """Random auxiliary feature batch, shape (BATCH_SIZE, 5)."""
    torch.manual_seed(1)
    return torch.randn(BATCH_SIZE, AUX_DIM)


@pytest.fixture
def model_default():
    """ExoplanetViT with default settings (pretrained=False, regression_head=True)."""
    from src.models.vit_model import ExoplanetViT

    return ExoplanetViT(pretrained=False)


@pytest.fixture
def model_no_regression():
    """ExoplanetViT with regression_head=False."""
    from src.models.vit_model import ExoplanetViT

    return ExoplanetViT(pretrained=False, regression_head=False)


@pytest.fixture
def model_frozen():
    """ExoplanetViT with freeze_backbone=True."""
    from src.models.vit_model import ExoplanetViT

    return ExoplanetViT(pretrained=False, freeze_backbone=True)


# ---------------------------------------------------------------------------
# 1. TestExoplanetViTOutputShape
# ---------------------------------------------------------------------------


class TestExoplanetViTOutputShape:
    """Forward pass produces tensors with the exact shapes specified in the API."""

    def test_class_logits_shape(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """class_logits must be (B, 2) for binary planet/non-planet classification."""
        model_default.eval()
        with torch.no_grad():
            class_logits, _ = model_default(image_batch, aux_batch)
        assert class_logits.shape == (BATCH_SIZE, 2), (
            f"Expected class_logits shape ({BATCH_SIZE}, 2), got {class_logits.shape}"
        )

    def test_reg_preds_shape(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """When regression_head=True, reg_preds must be (B, 3)."""
        model_default.eval()
        with torch.no_grad():
            _, reg_preds = model_default(image_batch, aux_batch)
        assert reg_preds is not None, "reg_preds should not be None when regression_head=True"
        assert reg_preds.shape == (BATCH_SIZE, 3), (
            f"Expected reg_preds shape ({BATCH_SIZE}, 3), got {reg_preds.shape}"
        )

    def test_no_regression_head(
        self,
        model_no_regression: nn.Module,
        image_batch: torch.Tensor,
        aux_batch: torch.Tensor,
    ) -> None:
        """When regression_head=False, the second return value must be None."""
        model_no_regression.eval()
        with torch.no_grad():
            _, reg_preds = model_no_regression(image_batch, aux_batch)
        assert reg_preds is None, (
            f"Expected reg_preds to be None when regression_head=False, got {reg_preds}"
        )


# ---------------------------------------------------------------------------
# 2. TestExoplanetViTInputHandling
# ---------------------------------------------------------------------------


class TestExoplanetViTInputHandling:
    """Model accepts correctly shaped inputs without raising exceptions."""

    def test_2channel_input(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """2-channel image input (B, 2, 224, 224) must not raise any exception."""
        assert image_batch.shape == (BATCH_SIZE, 2, IMG_H, IMG_W), (
            f"Fixture image_batch has unexpected shape {image_batch.shape}"
        )
        model_default.eval()
        with torch.no_grad():
            # Should complete without raising
            model_default(image_batch, aux_batch)

    def test_aux_tensor_input(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """Auxiliary tensor (B, 5) must be accepted without raising any exception."""
        assert aux_batch.shape == (BATCH_SIZE, AUX_DIM), (
            f"Fixture aux_batch has unexpected shape {aux_batch.shape}"
        )
        model_default.eval()
        with torch.no_grad():
            model_default(image_batch, aux_batch)


# ---------------------------------------------------------------------------
# 3. TestExoplanetViTFreezeBackbone
# ---------------------------------------------------------------------------


class TestExoplanetViTFreezeBackbone:
    """freeze_backbone=True freezes the ViT backbone while heads remain trainable."""

    def test_backbone_frozen(self, model_frozen: nn.Module) -> None:
        """All parameters belonging to the ViT backbone must have requires_grad=False."""
        frozen_params = [
            name
            for name, param in model_frozen.named_parameters()
            if name.startswith("backbone.") and param.requires_grad
        ]
        assert len(frozen_params) == 0, (
            f"Backbone parameters still require grad: {frozen_params[:5]}"
        )

    def test_heads_trainable_when_frozen(self, model_frozen: nn.Module) -> None:
        """class_head and aux_mlp parameters must remain trainable when backbone is frozen."""
        trainable_heads: list[str] = []
        for name, param in model_frozen.named_parameters():
            if name.startswith("class_head.") or name.startswith("aux_mlp."):
                if param.requires_grad:
                    trainable_heads.append(name)

        # Every class_head and aux_mlp param must be trainable
        all_head_params = [
            name
            for name, _ in model_frozen.named_parameters()
            if name.startswith("class_head.") or name.startswith("aux_mlp.")
        ]
        non_trainable = set(all_head_params) - set(trainable_heads)
        assert len(non_trainable) == 0, (
            f"Head parameters unexpectedly frozen: {non_trainable}"
        )
        assert len(trainable_heads) > 0, (
            "No trainable head parameters found — class_head/aux_mlp may be missing"
        )


# ---------------------------------------------------------------------------
# 4. TestExoplanetViTGradients
# ---------------------------------------------------------------------------


class TestExoplanetViTGradients:
    """Gradients flow correctly through the unfrozen parts of the model."""

    def test_gradients_flow_classification(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """CrossEntropyLoss.backward() must not raise and must produce non-None grads."""
        model_default.train()
        class_logits, _ = model_default(image_batch, aux_batch)
        labels = torch.zeros(BATCH_SIZE, dtype=torch.long)
        loss = nn.CrossEntropyLoss()(class_logits, labels)
        loss.backward()  # must not raise

        params_with_none_grad = [
            name
            for name, param in model_default.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert len(params_with_none_grad) == 0, (
            f"Parameters still have None gradient after backward: {params_with_none_grad[:5]}"
        )

    def test_gradients_flow_regression(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """MSELoss on reg_preds.backward() must not raise."""
        model_default.train()
        _, reg_preds = model_default(image_batch, aux_batch)
        assert reg_preds is not None, "reg_preds is None — cannot test regression gradients"
        targets = torch.zeros(BATCH_SIZE, 3)
        loss = nn.MSELoss()(reg_preds, targets)
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# 5. TestExoplanetViTAttentionMaps
# ---------------------------------------------------------------------------


class TestExoplanetViTAttentionMaps:
    """get_attention_maps returns properly structured per-block attention weights."""

    def test_get_attention_maps_returns_dict(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """get_attention_maps must return a dict, not a list/tensor/None."""
        model_default.eval()
        with torch.no_grad():
            attn = model_default.get_attention_maps(image_batch, aux_batch)
        assert isinstance(attn, dict), (
            f"Expected dict from get_attention_maps, got {type(attn)}"
        )

    def test_attention_maps_all_blocks(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """Dict must have exactly 12 keys: 'block_0' through 'block_11'."""
        model_default.eval()
        with torch.no_grad():
            attn = model_default.get_attention_maps(image_batch, aux_batch)

        expected_keys = {f"block_{i}" for i in range(VIT_NUM_BLOCKS)}
        actual_keys = set(attn.keys())
        assert actual_keys == expected_keys, (
            f"Attention map keys mismatch.\n"
            f"  Missing : {expected_keys - actual_keys}\n"
            f"  Extra   : {actual_keys - expected_keys}"
        )

    def test_attention_map_shape(
        self, model_default: nn.Module, image_batch: torch.Tensor, aux_batch: torch.Tensor
    ) -> None:
        """Each attention tensor must be (B, 12, 197, 197) for ViT-B/16 with 224×224 input.

        Derivation
        ----------
        - patch_size = 16  →  num_patches = (224/16)^2 = 196
        - seq_len = num_patches + 1 (CLS token) = 197
        - num_heads = 12  (ViT-B/16 specification)
        """
        model_default.eval()
        with torch.no_grad():
            attn = model_default.get_attention_maps(image_batch, aux_batch)

        expected_shape = (BATCH_SIZE, VIT_NUM_HEADS, VIT_SEQ_LEN, VIT_SEQ_LEN)
        for key, tensor in attn.items():
            assert isinstance(tensor, torch.Tensor), (
                f"Attention map '{key}' is not a Tensor, got {type(tensor)}"
            )
            assert tensor.shape == expected_shape, (
                f"Attention map '{key}': expected shape {expected_shape}, got {tensor.shape}"
            )


# ---------------------------------------------------------------------------
# 6. TestAuxMLP
# ---------------------------------------------------------------------------


class TestAuxMLP:
    """AuxMLP maps (B, aux_dim) → (B, aux_out=32) correctly."""

    def test_aux_mlp_output_shape(self) -> None:
        """Input (B=4, 5) must produce output (B=4, 32)."""
        from src.models.vit_model import AuxMLP

        batch_size = 4
        aux_mlp = AuxMLP()
        aux_mlp.eval()

        x = torch.randn(batch_size, AUX_DIM)
        with torch.no_grad():
            out = aux_mlp(x)

        assert out.shape == (batch_size, 32), (
            f"Expected AuxMLP output shape ({batch_size}, 32), got {out.shape}"
        )
