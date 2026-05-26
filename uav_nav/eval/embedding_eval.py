"""Evaluation of the semantic embedding head quality.

Metrics:
    - ROC AUC: binary same-instance vs diff-instance classification by cosine sim
    - Recall@K: for each query, check if correct instance is in top-K by cosine sim
    - Discriminability: inter/intra class distance ratio

Can run as a standalone script after training::

    python -m uav_nav.eval.embedding_eval ^
        --weights weights/embedding_head.pt ^
        --index_dir data/instance_index ^
        --split val ^
        --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger

from uav_nav.perception.embedding_head import EmbeddingHead
from uav_nav.data.triplet_dataset import _extract_crop, _default_transform


# ── metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class EmbeddingMetrics:
    """Descriptor quality metrics.

    Attributes:
        roc_auc: AUC of ROC curve for same/diff instance binary task.
        recall_at_1: Fraction of queries where correct instance is rank-1.
        recall_at_5: Fraction of queries where correct instance is in top-5.
        intra_dist_mean: Mean cosine distance within the same instance.
        inter_dist_mean: Mean cosine distance between different instances (same class).
        discriminability: inter_dist / intra_dist (higher is better).
        n_queries: Number of query embeddings evaluated.
        n_instances: Number of unique instances in gallery.
    """
    roc_auc: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    intra_dist_mean: float = 0.0
    inter_dist_mean: float = 0.0
    discriminability: float = 0.0
    n_queries: int = 0
    n_instances: int = 0


# ── core math ─────────────────────────────────────────────────────────────────

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    idx = np.argsort(-scores)
    sl = labels[idx].astype(float)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(sl)
    fp = np.cumsum(1.0 - sl)
    tpr, fpr = tp / n_pos, fp / n_neg
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(tpr, fpr))


def _recall_at_k(
    query_embs: np.ndarray,      # (Q, D)
    query_iids: list[str],
    gallery_embs: np.ndarray,    # (G, D)
    gallery_iids: list[str],
    k: int,
) -> float:
    """Fraction of queries where correct instance_id appears in top-K gallery."""
    # Cosine similarity (embeddings are L2-normalised)
    sims = query_embs @ gallery_embs.T  # (Q, G)
    hits = 0
    for qi, q_iid in enumerate(query_iids):
        top_k_idx = np.argpartition(-sims[qi], min(k, len(gallery_iids) - 1))[:k]
        top_k_iids = {gallery_iids[i] for i in top_k_idx}
        if q_iid in top_k_iids:
            hits += 1
    return hits / len(query_iids) if query_iids else 0.0


# ── embedding extraction ──────────────────────────────────────────────────────

def _embed_split(
    csv_path: Path,
    head: EmbeddingHead,
    device: str,
    batch_size: int = 128,
    max_instances: Optional[int] = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Embed one observation per instance from the CSV.

    Returns:
        embeddings: (N, D) float32
        instance_ids: list[str] length N
        class_ids: list[int] length N
    """
    transform = _default_transform(head.config.crop_size)
    crop_size = head.config.crop_size
    pad_ratio  = head.config.crop_pad_ratio

    # Load CSV, group by instance_id, keep first 2 rows per instance
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    iid_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        iid_to_rows[row["instance_id"]].append(row)

    usable = [(iid, rs) for iid, rs in iid_to_rows.items() if len(rs) >= 2]
    if max_instances:
        usable = usable[:max_instances]

    logger.info("  Embedding {} instances from {}", len(usable), csv_path.name)

    crops_buf: list[torch.Tensor] = []
    query_iids: list[str] = []
    gallery_iids: list[str] = []
    query_class: list[int] = []
    gallery_class: list[int] = []

    for iid, rs in usable:
        r0, r1 = rs[0], rs[1]
        for row, tgt_iids, tgt_class in [(r0, query_iids, query_class),
                                          (r1, gallery_iids, gallery_class)]:
            img = cv2.imread(row["image_path"])
            if img is None:
                crop_arr = np.full((crop_size, crop_size, 3), 127, dtype=np.uint8)
            else:
                crop_arr = _extract_crop(img, row["polygon_flat"], crop_size, pad_ratio)
                if crop_arr is None:
                    crop_arr = np.full((crop_size, crop_size, 3), 127, dtype=np.uint8)
            crops_buf.append(transform(crop_arr))
            tgt_iids.append(iid)
            tgt_class.append(int(row["class_id"]))

    # Run in batches
    all_embs: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(crops_buf), batch_size):
            batch = torch.stack(crops_buf[start:start + batch_size]).to(device)
            all_embs.append(head(batch).cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)  # (2*N, D)
    # Split back into query and gallery halves
    n = len(usable)
    query_embs   = embs[:n]
    gallery_embs = embs[n:]

    return query_embs, gallery_embs, query_iids, gallery_iids, query_class, gallery_class


# ── evaluator ─────────────────────────────────────────────────────────────────

class EmbeddingEvaluator:
    """Evaluates descriptor quality via retrieval on a labelled gallery.

    Args:
        distance_metric: ``"cosine"`` (default) — embeddings must be L2-normalised.
    """

    def __init__(self, distance_metric: str = "cosine") -> None:
        self.distance_metric = distance_metric

    def evaluate(
        self,
        query_embs: np.ndarray,
        gallery_embs: np.ndarray,
        query_iids: list[str],
        gallery_iids: list[str],
        query_class: list[int],
        gallery_class: list[int],
    ) -> EmbeddingMetrics:
        """Compute all metrics.

        Args:
            query_embs: L2-normalised query embeddings (N, D).
            gallery_embs: L2-normalised gallery embeddings (N, D).
            query_iids: Instance IDs for each query row.
            gallery_iids: Instance IDs for each gallery row.
            query_class: Class IDs for queries.
            gallery_class: Class IDs for gallery rows.

        Returns:
            EmbeddingMetrics with all fields filled.
        """
        sims = query_embs @ gallery_embs.T  # (N, N), cosine

        # ROC AUC: pos = same instance, neg = different instance same class
        pos_sims, neg_sims = [], []
        for i, (q_iid, q_cls) in enumerate(zip(query_iids, query_class)):
            for j, (g_iid, g_cls) in enumerate(zip(gallery_iids, gallery_class)):
                if i == j:
                    continue
                if g_cls != q_cls:
                    continue
                if q_iid == g_iid:
                    pos_sims.append(float(sims[i, j]))
                else:
                    neg_sims.append(float(sims[i, j]))

        scores = np.array(pos_sims + neg_sims)
        labels = np.array([1] * len(pos_sims) + [0] * len(neg_sims))
        auc = _roc_auc(scores, labels)

        # Recall@K (leave-self-out: don't include query itself in gallery)
        r1 = _recall_at_k(query_embs, query_iids, gallery_embs, gallery_iids, k=1)
        r5 = _recall_at_k(query_embs, query_iids, gallery_embs, gallery_iids, k=5)

        # Intra/inter class distance
        intra = np.mean([1 - float(sims[i, i]) for i in range(len(query_iids))])
        same_cls_pairs = [
            1 - float(sims[i, j])
            for i in range(len(query_iids))
            for j in range(len(gallery_iids))
            if i != j and query_class[i] == gallery_class[j] and query_iids[i] != gallery_iids[j]
        ]
        inter = float(np.mean(same_cls_pairs)) if same_cls_pairs else 0.0
        disc = inter / intra if intra > 0 else 0.0

        return EmbeddingMetrics(
            roc_auc=auc,
            recall_at_1=r1,
            recall_at_5=r5,
            intra_dist_mean=intra,
            inter_dist_mean=inter,
            discriminability=disc,
            n_queries=len(query_iids),
            n_instances=len(set(query_iids)),
        )


# ── standalone script ─────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> EmbeddingMetrics:
    head = EmbeddingHead.load(args.weights, device=args.device)
    csv_path = Path(args.index_dir) / f"{args.split}.csv"

    q_embs, g_embs, q_iids, g_iids, q_cls, g_cls = _embed_split(
        csv_path, head, args.device,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
    )

    evaluator = EmbeddingEvaluator()
    metrics = evaluator.evaluate(q_embs, g_embs, q_iids, g_iids, q_cls, g_cls)

    logger.info("── Embedding Eval ({}) ──────────────────", args.split)
    logger.info("  ROC AUC:         {:.4f}", metrics.roc_auc)
    logger.info("  Recall@1:        {:.4f}", metrics.recall_at_1)
    logger.info("  Recall@5:        {:.4f}", metrics.recall_at_5)
    logger.info("  Intra dist:      {:.4f}", metrics.intra_dist_mean)
    logger.info("  Inter dist:      {:.4f}", metrics.inter_dist_mean)
    logger.info("  Discriminability:{:.4f}", metrics.discriminability)
    logger.info("  Instances:       {}", metrics.n_instances)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(metrics), indent=2))
        logger.success("Saved → {}", out)

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EmbeddingHead retrieval quality")
    p.add_argument("--weights",       required=True,  help="Path to embedding_head.pt")
    p.add_argument("--index_dir",     default="data/instance_index")
    p.add_argument("--split",         default="val",  choices=["train", "val", "test"])
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--max_instances", type=int, default=None,
                   help="Limit number of instances (for quick sanity checks)")
    p.add_argument("--out",           default=None,   help="Save JSON metrics to this path")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(_parse_args())
