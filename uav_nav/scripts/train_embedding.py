"""Train the semantic embedding head via TripletMarginLoss.

Usage::

    python -m uav_nav.scripts.train_embedding
    python -m uav_nav.scripts.train_embedding --device cuda --epochs 50 --batch_size 64
    python -m uav_nav.scripts.train_embedding --resume experiments/20260526_emb/best.pt

Reads from:
    data/instance_index/train.csv, val.csv

Writes to:
    experiments/<date>_embedding/
        checkpoints/epoch_NN.pt
        best.pt
        train_log.csv
    weights/embedding_head.pt   (symlink/copy of best)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from uav_nav.data.triplet_dataset import TripletDataset, build_loaders
from uav_nav.perception.embedding_head import EmbeddingConfig, EmbeddingHead


# ── ROC AUC (no sklearn required) ────────────────────────────────────────────

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC for binary classification."""
    idx = np.argsort(-scores)
    sorted_labels = labels[idx].astype(float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # np.trapz removed in NumPy 2.0; np.trapezoid is the replacement
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(tpr, fpr))


# ── per-epoch eval ────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(head: EmbeddingHead, loader, device: str, max_batches: int = 200) -> dict[str, float]:
    """Compute triplet loss and ROC AUC on validation set.

    Returns dict with keys: val_loss, val_auc.
    """
    head.eval()
    criterion = nn.TripletMarginLoss(margin=head.config.margin, p=2)
    losses: list[float] = []
    pos_sims: list[float] = []
    neg_sims: list[float] = []

    for step, (a, p, n) in enumerate(loader):
        if step >= max_batches:
            break
        a, p, n = a.to(device), p.to(device), n.to(device)
        ea, ep, en = head(a), head(p), head(n)
        losses.append(criterion(ea, ep, en).item())
        pos_sims.extend((ea * ep).sum(dim=1).cpu().tolist())
        neg_sims.extend((ea * en).sum(dim=1).cpu().tolist())

    pos_sims_arr = np.array(pos_sims)
    neg_sims_arr = np.array(neg_sims)
    scores = np.concatenate([pos_sims_arr, neg_sims_arr])
    labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    auc = _roc_auc(scores, labels)

    head.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_auc": auc,
    }


# ── training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    run_dir = Path("experiments") / f"{datetime.now():%Y%m%d_%H%M%S}_embedding"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    log_path = run_dir / "train_log.csv"

    logger.add(run_dir / "train.log", level="DEBUG")
    logger.info("Run dir: {}", run_dir)

    # ── dataset ───────────────────────────────────────────────────────────────
    index_dir = Path(args.index_dir)
    logger.info("Loading instance index from {}", index_dir)
    train_loader, val_loader = build_loaders(
        index_dir=index_dir,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Train: {} instances → {} batches/epoch | Val: {} instances",
        len(train_loader.dataset),
        len(train_loader),
        len(val_loader.dataset),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    cfg = EmbeddingConfig(
        embedding_dim=args.embedding_dim,
        backbone_channels=576,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        margin=args.margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_backbone_epochs=args.freeze_epochs,
        crop_size=args.crop_size,
    )
    head = EmbeddingHead(config=cfg, device=args.device, pretrained=True)

    if args.resume:
        logger.info("Resuming from {}", args.resume)
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        head.load_state_dict(ckpt["state_dict"])

    head.freeze_backbone()
    logger.info("Backbone frozen for first {} epochs", args.freeze_epochs)

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    def _make_optimizer() -> torch.optim.Optimizer:
        params = [
            {"params": head.head.parameters(), "lr": args.lr},
        ]
        if not all(p.requires_grad for p in head.backbone.parameters()):
            pass  # backbone frozen — only head params
        else:
            params.append({"params": head.backbone.parameters(), "lr": args.lr * 0.1})
        return torch.optim.AdamW(params, weight_decay=args.weight_decay)

    optimizer = _make_optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    # ── log header ────────────────────────────────────────────────────────────
    log_fields = ["epoch", "train_loss", "val_loss", "val_auc", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_auc = -1.0
    best_ckpt = run_dir / "best.pt"

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # Unfreeze backbone after warmup
        if epoch == args.freeze_epochs + 1:
            head.unfreeze_backbone()
            optimizer = _make_optimizer()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )
            logger.info("Epoch {}: backbone unfrozen, optimizer rebuilt", epoch)

        # ── train step ────────────────────────────────────────────────────────
        head.train()
        epoch_losses: list[float] = []
        for a, p, n in train_loader:
            a, p, n = a.to(args.device), p.to(args.device), n.to(args.device)
            ea, ep, en = head(a), head(p), head(n)
            loss = criterion(ea, ep, en)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        train_loss = float(np.mean(epoch_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        # ── val step ──────────────────────────────────────────────────────────
        val_metrics = _evaluate(head, val_loader, args.device)
        val_loss = val_metrics["val_loss"]
        val_auc  = val_metrics["val_auc"]

        logger.info(
            "Epoch {:3d}/{} | train_loss={:.4f} | val_loss={:.4f} | val_AUC={:.4f} | lr={:.2e}",
            epoch, args.epochs, train_loss, val_loss, val_auc, current_lr,
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            head.save(ckpt_dir / f"epoch_{epoch:03d}.pt")

        if val_auc > best_auc or math.isnan(best_auc):
            best_auc = val_auc
            head.save(best_ckpt)
            logger.success("  New best AUC={:.4f} → {}", best_auc, best_ckpt)

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 6),
                "lr": round(current_lr, 8),
            })

    # ── save final ────────────────────────────────────────────────────────────
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    final_path = weights_dir / "embedding_head.pt"
    head.save(final_path)
    logger.success("Final weights → {}", final_path)
    logger.success("Best AUC={:.4f} → {}", best_auc, best_ckpt)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmbeddingHead via TripletMarginLoss")
    p.add_argument("--index_dir",     default="data/instance_index")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--margin",        type=float, default=0.5)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--embedding_dim", type=int,   default=128)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--crop_size",     type=int,   default=96)
    p.add_argument("--freeze_epochs", type=int,   default=5,
                   help="Epochs to keep backbone frozen")
    p.add_argument("--num_workers",   type=int,   default=0,
                   help="DataLoader workers (0 = main process; safer on Windows)")
    p.add_argument("--save_every",    type=int,   default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume",        default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
