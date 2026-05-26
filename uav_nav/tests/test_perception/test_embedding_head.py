"""Unit tests for EmbeddingHead.

All tests run on CPU without the MidAir dataset.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from uav_nav.perception.embedding_head import EmbeddingConfig, EmbeddingHead


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> EmbeddingConfig:
    return EmbeddingConfig(embedding_dim=16, hidden_dim=32, backbone_channels=576)


@pytest.fixture(scope="module")
def head(cfg: EmbeddingConfig) -> EmbeddingHead:
    return EmbeddingHead(config=cfg, device="cpu", pretrained=False)


# ── forward shape & L2 norm ───────────────────────────────────────────────────

class TestForward:
    def test_output_shape(self, head: EmbeddingHead, cfg: EmbeddingConfig) -> None:
        x = torch.zeros(4, 3, 96, 96)
        out = head(x)
        assert out.shape == (4, cfg.embedding_dim)

    def test_l2_normalized(self, head: EmbeddingHead) -> None:
        x = torch.randn(8, 3, 96, 96)
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_sample(self, head: EmbeddingHead, cfg: EmbeddingConfig) -> None:
        head.eval()
        x = torch.randn(1, 3, 96, 96)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (1, cfg.embedding_dim)

    def test_gradients_flow(self, head: EmbeddingHead) -> None:
        # Verify backprop works end-to-end through the projection head.
        # Using the shared pretrained=False model in train mode (BN uses batch stats
        # rather than potentially-stale running stats from earlier test passes).
        head.train()
        head.unfreeze_backbone()
        x = torch.randn(4, 3, 96, 96)
        out = head(x)
        out.sum().backward()
        head_grads = [p.grad for p in head.head.parameters() if p.grad is not None]
        assert head_grads, "No gradients in head"
        assert any(g.abs().max() > 0 for g in head_grads), "All head gradients are zero"


# ── encode_crop ───────────────────────────────────────────────────────────────

class TestEncodeCrop:
    def _square_polygon(self, cx: float, cy: float, r: float = 0.1) -> list[float]:
        return [cx-r, cy-r, cx+r, cy-r, cx+r, cy+r, cx-r, cy+r]

    def test_output_shape(self, head: EmbeddingHead, cfg: EmbeddingConfig) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        poly = self._square_polygon(0.5, 0.5)
        emb = head.encode_crop(img, poly)
        assert emb.shape == (cfg.embedding_dim,)

    def test_output_l2_normalized(self, head: EmbeddingHead) -> None:
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        poly = self._square_polygon(0.5, 0.5)
        emb = head.encode_crop(img, poly)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_degenerate_bbox_returns_zeros(self, head: EmbeddingHead, cfg: EmbeddingConfig) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # Polygon with 0 width
        poly = [0.5, 0.5, 0.5, 0.5]
        emb = head.encode_crop(img, poly)
        assert emb.shape == (cfg.embedding_dim,)
        assert np.allclose(emb, 0.0)

    def test_mask_fills_background(self, head: EmbeddingHead) -> None:
        # Create image with known color, polygon covering half
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        # Triangle in top-left quarter
        poly = [0.0, 0.0, 0.25, 0.0, 0.0, 0.25]
        emb = head.encode_crop(img, poly)
        assert emb is not None
        assert emb.shape[0] > 0


# ── freeze / unfreeze ─────────────────────────────────────────────────────────

class TestFreezeBackbone:
    def test_freeze_sets_no_grad(self, head: EmbeddingHead) -> None:
        head.freeze_backbone()
        for p in head.backbone.parameters():
            assert not p.requires_grad

    def test_unfreeze_restores_grad(self, head: EmbeddingHead) -> None:
        head.freeze_backbone()
        head.unfreeze_backbone()
        assert any(p.requires_grad for p in head.backbone.parameters())


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path, cfg: EmbeddingConfig) -> None:
        model = EmbeddingHead(config=cfg, device="cpu", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            before = model(x).clone()

        ckpt = tmp_path / "emb.pt"
        model.save(ckpt)

        loaded = EmbeddingHead.load(ckpt, device="cpu")  # load() sets eval()
        with torch.no_grad():
            after = loaded(x)

        assert torch.allclose(before, after, atol=1e-6)
        assert loaded.config.embedding_dim == cfg.embedding_dim

    def test_load_sets_eval_mode(self, tmp_path: Path, cfg: EmbeddingConfig) -> None:
        model = EmbeddingHead(config=cfg, device="cpu", pretrained=False)
        model.save(tmp_path / "emb2.pt")
        loaded = EmbeddingHead.load(tmp_path / "emb2.pt")
        assert not loaded.training


# ── build() shim ─────────────────────────────────────────────────────────────

class TestLegacyShims:
    def test_build_is_noop(self, head: EmbeddingHead) -> None:
        head.build()  # must not raise

    def test_train_epoch_raises(self, head: EmbeddingHead) -> None:
        with pytest.raises(NotImplementedError):
            head.train_epoch(None, None)

    def test_evaluate_raises(self, head: EmbeddingHead) -> None:
        with pytest.raises(NotImplementedError):
            head.evaluate(None)
