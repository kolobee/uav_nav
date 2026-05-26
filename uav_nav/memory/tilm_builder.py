"""TILM construction from a sequence of landmark observations.

TILMBuilder processes a stream of frames (with poses and landmarks) and
incrementally builds the Topological Invariant Landmark Map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uav_nav.memory.tilm import TILM, TILMNode, TILMEdge
from uav_nav.perception.landmark_extractor import Landmark


@dataclass
class BuilderConfig:
    """Configuration for the TILM building process.

    Attributes:
        min_node_distance: Minimum travel distance (m) to create a new node.
        min_landmarks_per_node: Discard nodes with fewer than this many
            landmarks.
        keyframe_overlap_threshold: Maximum fraction of shared landmarks with
            the previous keyframe before a new node is forced.
        edge_max_distance: Maximum distance (m) to add an edge between nodes.
        use_loop_closure: Whether to detect and add loop closure edges.
        loop_closure_threshold: Minimum descriptor similarity for loop closure.
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

        A new node is created if the UAV has moved at least
        ``config.min_node_distance`` metres since the last node and the
        current frame has at least ``config.min_landmarks_per_node`` valid
        landmarks.

        Args:
            pose_ned: Current NED pose as 4x4 SE(3) matrix, float64.
            landmarks: Landmark observations for the current frame.
            timestamp: Frame timestamp in seconds.
            keyframe_id: Optional keyframe identifier string.

        Returns:
            The new node ID if a node was created, otherwise None.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILMBuilder.process_frame is not yet implemented.")

    def _should_create_node(
        self, position: np.ndarray, landmarks: list[Landmark]
    ) -> bool:
        """Decide whether to create a new node at the current position.

        Args:
            position: Current NED position, shape (3,).
            landmarks: Current landmark observations.

        Returns:
            True if a new node should be created.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILMBuilder._should_create_node is not yet implemented.")

    def _try_loop_closure(self, new_node_id: int) -> list[TILMEdge]:
        """Search for loop closure candidates and create edges.

        Args:
            new_node_id: ID of the newly created node.

        Returns:
            List of newly created loop closure TILMEdge instances.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILMBuilder._try_loop_closure is not yet implemented.")

    def finalise(self) -> TILM:
        """Finalise the map and return it.

        Runs any post-processing steps (e.g. descriptor index building,
        compaction, statistics logging).

        Returns:
            The completed TILM instance.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILMBuilder.finalise is not yet implemented.")
