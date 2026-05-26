"""TILM construction from a sequence of landmark observations.

TILMBuilder processes a stream of frames (with poses and landmarks) and
incrementally builds the Topological Invariant Landmark Map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uav_nav.memory.tilm import TILM, TILMEdge, TILMNode
from uav_nav.perception.landmark_extractor import Landmark


@dataclass
class BuilderConfig:
    """Configuration for the TILM building process.

    Attributes:
        min_node_distance: Minimum travel distance (m) to create a new node.
        min_landmarks_per_node: Discard nodes with fewer than this many landmarks.
        edge_max_distance: Maximum distance (m) to add a loop-closure edge.
        use_loop_closure: Whether to detect and add loop closure edges.
        loop_closure_threshold: Minimum place-descriptor cosine similarity
            for loop closure (0–1).
    """

    min_node_distance: float = 5.0
    min_landmarks_per_node: int = 2
    keyframe_overlap_threshold: float = 0.5
    edge_max_distance: float = 20.0
    use_loop_closure: bool = True
    loop_closure_threshold: float = 0.75


class TILMBuilder:
    """Incrementally constructs a TILM from landmark observations.

    Args:
        tilm: TILM instance to populate.
        config: BuilderConfig controlling node/edge creation heuristics.
    """

    def __init__(self, tilm: TILM, config: Optional[BuilderConfig] = None) -> None:
        self.tilm = tilm
        self.config = config or BuilderConfig()
        self._prev_position: Optional[np.ndarray] = None
        self._prev_node_id: Optional[int] = None
        self._frame_count: int = 0

    def process_frame(
        self,
        pose_ned: np.ndarray,
        landmarks: list[Landmark],
        timestamp: float,
        keyframe_id: str = "",
    ) -> Optional[int]:
        """Process one frame and optionally create a new TILM node.

        A new node is created when:
        - The UAV has moved at least ``min_node_distance`` metres, AND
        - The frame has at least ``min_landmarks_per_node`` landmarks.

        A sequential edge is added from the previous node to the new node.
        Loop closure edges are attempted if enabled.

        Args:
            pose_ned: Current NED pose as 4×4 SE(3) matrix, float64.
            landmarks: Landmark observations for the current frame.
            timestamp: Frame timestamp in seconds.
            keyframe_id: Optional keyframe identifier string.

        Returns:
            New node ID if a node was created, otherwise None.
        """
        self._frame_count += 1
        position = np.asarray(pose_ned[:3, 3], dtype=np.float64)

        if not self._should_create_node(position, landmarks):
            if self._prev_position is None:
                self._prev_position = position.copy()
            return None

        # ── Create node ───────────────────────────────────────────────────────
        node_id = self.tilm.n_nodes()
        place_desc = _mean_descriptor(landmarks)
        node = TILMNode(
            node_id=node_id,
            position_ned=position.copy(),
            landmarks=list(landmarks),
            place_descriptor=place_desc,
            timestamp=timestamp,
            keyframe_id=keyframe_id,
        )
        self.tilm.add_node(node)

        # ── Sequential edge from previous node ────────────────────────────────
        if self._prev_node_id is not None and self._prev_position is not None:
            dist = float(np.linalg.norm(position - self._prev_position))
            self.tilm.add_edge(TILMEdge(
                src_id=self._prev_node_id,
                dst_id=node_id,
                relative_pose=np.eye(4, dtype=np.float64),
                distance=dist,
            ))

        self._prev_node_id = node_id
        self._prev_position = position.copy()

        # ── Loop closure ──────────────────────────────────────────────────────
        if self.config.use_loop_closure:
            for edge in self._try_loop_closure(node_id):
                self.tilm.add_edge(edge)

        return node_id

    def _should_create_node(
        self, position: np.ndarray, landmarks: list[Landmark]
    ) -> bool:
        """Decide whether to create a new node at the current position."""
        if len(landmarks) < self.config.min_landmarks_per_node:
            return False
        if self._prev_position is None:
            return True
        dist = float(np.linalg.norm(position - self._prev_position))
        return dist >= self.config.min_node_distance

    def _try_loop_closure(self, new_node_id: int) -> list[TILMEdge]:
        """Search for loop closure candidates and create edges.

        Criteria:
        1. Node is within ``edge_max_distance`` of the new node.
        2. Place-descriptor cosine similarity ≥ ``loop_closure_threshold``.
        3. Not the immediately preceding sequential node.
        """
        new_node = self.tilm.get_node(new_node_id)
        if new_node.place_descriptor is None:
            return []

        edges: list[TILMEdge] = []
        for node in self.tilm.all_nodes:
            if node.node_id == new_node_id:
                continue
            if node.node_id == self._prev_node_id:
                continue
            if node.place_descriptor is None:
                continue

            dist = float(np.linalg.norm(node.position_ned - new_node.position_ned))
            if dist < 1.0 or dist > self.config.edge_max_distance:
                continue

            sim = float(np.dot(new_node.place_descriptor, node.place_descriptor))
            if sim >= self.config.loop_closure_threshold:
                edges.append(TILMEdge(
                    src_id=new_node_id,
                    dst_id=node.node_id,
                    relative_pose=np.eye(4, dtype=np.float64),
                    distance=dist,
                    weight=max(1.0 - sim, 1e-6),
                ))
        return edges

    def finalise(self) -> TILM:
        """Finalise and return the completed TILM.

        Returns:
            The populated TILM instance.
        """
        return self.tilm


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean_descriptor(landmarks: list[Landmark]) -> Optional[np.ndarray]:
    """Compute the L2-normalised mean embedding over a set of landmarks."""
    if not landmarks:
        return None
    descs = np.stack([lm.descriptor for lm in landmarks])   # (N, D)
    mean = descs.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-9 else None
