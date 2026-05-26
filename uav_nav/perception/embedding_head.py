"""Embedding head for semantic landmark descriptors.

MobileNetV3-Small backbone with a two-layer projection head trained via
TripletMarginLoss.  Produces L2-normalised 128-d descriptors from masked
96×96 image crops of segmented landmark instances.

Training workflow:
    1. Run ``build_instance_index.py`` to obtain per-split CSV tables.
    2. Train via ``scripts/train_embedding.py``.
    3. Export with ``deployment/export_onnx.py``.

Inference:
    Use ``encode_crop`` to embed a single detection given the raw BGR image
    and its YOLO polygon (flat normalised coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """Hyperparameters for EmbeddingHead.

    Attributes:
        embedding_dim: Output descriptor dimensionality.
        backbone_channels: Feature channels from MobileNetV3-Small pool layer.
        hidden_dim: Intermediate projection dimension.
        dropout: Dropout probability in projection head.
        margin: Triplet loss margin.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        epochs: Total training epochs.
        batch_size: Triplet batch size (number of anchors per batch).
        freeze_backbone_epochs: Epochs to keep backbone frozen at training start.
        crop_size: Spatial size of masked crops fed to the network.
        crop_pad_ratio: Fractional padding added around the tight bbox.
    """

    embedding_dim: int = 128
    backbone_channels: int = 576       # MobileNetV3-Small AdaptiveAvgPool output
    hidden_dim: int = 256
    dropout: float = 0.1
    margin: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 64
    freeze_backbone_epochs: int = 5
    crop_size: int = 96
    crop_pad_ratio: float = 0.20
    temperature: float = 0.07  # for NT-Xent if used instead of triplet


# ── model ─────────────────────────────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """MobileNetV3-Small + projection head producing L2-normalised descriptors.

    Args:
        config: Embedding hyperparameters.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        pretrained: Whether to initialise backbone with ImageNet weights.
    """

    # ImageNet statistics used for normalisation
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        device: str = "cpu",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or EmbeddingConfig()
        self.device_str = device

        # Backbone: MobileNetV3-Small, features + avg-pool (no classifier)
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        _backbone = models.mobilenet_v3_small(weights=weights)
        self.backbone: nn.Sequential = _backbone.features
        self.pool: nn.Module = _backbone.avgpool  # AdaptiveAvgPool2d(1)

        cfg = self.config
        self.head = nn.Sequential(
            nn.Linear(cfg.backbone_channels, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),  # works with batch_size=1 in any mode
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

        self.to(device)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Embed a batch of masked crops.

        Args:
            crops: Float32 tensor ``(B, 3, crop_size, crop_size)`` normalised
                with ImageNet mean/std.

        Returns:
            L2-normalised descriptor matrix ``(B, embedding_dim)``.
        """
        x = self.backbone(crops)          # (B, 576, H', W')
        x = self.pool(x).flatten(1)      # (B, 576)
        x = self.head(x)                 # (B, 128)
        return F.normalize(x, dim=1)

    # ── backbone freezing ─────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (call at epoch 0 of training)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ── crop inference ────────────────────────────────────────────────────────

    def encode_crop(
        self,
        img_bgr: np.ndarray,
        polygon_flat: list[float],
    ) -> np.ndarray:
        """Extract a masked crop and return its L2-normalised embedding.

        Args:
            img_bgr: Full-resolution BGR image as a uint8 numpy array.
            polygon_flat: Flat list of normalised polygon coordinates
                ``[x0, y0, x1, y1, …]`` in ``[0, 1]``.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``.  Returns a
            zero vector if the bounding box is degenerate (< 4 px side).
        """
        h, w = img_bgr.shape[:2]
        cfg = self.config

        coords = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2)
        xs, ys = coords[:, 0], coords[:, 1]

        # Tight bbox → padded bbox (clamped to image)
        pad_x = (xs.max() - xs.min()) * cfg.crop_pad_ratio
        pad_y = (ys.max() - ys.min()) * cfg.crop_pad_ratio
        x1 = max(0.0, xs.min() - pad_x); x2 = min(1.0, xs.max() + pad_x)
        y1 = max(0.0, ys.min() - pad_y); y2 = min(1.0, ys.max() + pad_y)

        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        if px2 - px1 < 4 or py2 - py1 < 4:
            return np.zeros(cfg.embedding_dim, dtype=np.float32)

        crop = img_bgr[py1:py2, px1:px2].copy()

        # Polygon mask — fill background with neutral gray
        pts_px = (coords * np.array([w, h])).astype(np.int32)
        pts_local = pts_px - np.array([px1, py1])
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_local], 255)
        bg = np.uint8(127)
        for c in range(3):
            crop[..., c] = np.where(mask > 0, crop[..., c], bg)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (cfg.crop_size, cfg.crop_size),
                                  interpolation=cv2.INTER_LINEAR)

        tensor = self._transform(crop_resized).unsqueeze(0).to(self.device_str)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                emb = self.forward(tensor)
        finally:
            if was_training:
                self.train()
        return emb.squeeze(0).cpu().numpy()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save model weights and config to a single ``.pt`` checkpoint.

        Args:
            path: Destination file path.  Parent directories are created if
                they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "EmbeddingHead":
        """Load a checkpoint saved by :meth:`save`.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            device: Target device string.

        Returns:
            EmbeddingHead in eval mode on the requested device.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        head = cls(config=ckpt["config"], device=device, pretrained=False)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        return head

    # ── legacy shims (kept for backward-compat with old train script) ─────────

    def build(self) -> None:
        """No-op: model is built in ``__init__``.  Kept for compatibility."""

    def train_epoch(self, dataloader: object, optimizer: object) -> float:
        """Run one training epoch.  Implemented in ``scripts/train_embedding.py``.

        Raises:
            NotImplementedError: Call the training script instead.
        """
        raise NotImplementedError("Use scripts/train_embedding.py for training.")

    def evaluate(self, dataloader: object) -> dict[str, float]:
        """Evaluate descriptor quality.  Implemented in ``eval/embedding_eval.py``.

        Raises:
            NotImplementedError: Call the eval module instead.
        """
        raise NotImplementedError("Use eval/embedding_eval.py for evaluation.")
