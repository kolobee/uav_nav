"""Topological Invariant Landmark Map (TILM).

The TILM is a topological graph where each node represents a geographic
location annotated with a set of semantic landmarks observed from that
position. Edges encode traversability and relative pose constraints.

The map is weather-invariant because it uses semantic and geometric
descriptors rather than raw appearance features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uav_nav.perception.landmark_extractor import Landmark


@dataclass
class TILMNode:
    """A node in the TILM topological graph.

    Attributes:
        node_id: Unique integer node identifier.
        position_ned: NED position estimate when the node was created,
            shape (3,), float64. May be approximate.
        landmarks: Semantic landmarks observed from this location.
        place_descriptor: Compact holistic descriptor for place recognition.
        timestamp: Creation timestamp in seconds.
        keyframe_id: Identifier of the source keyframe.
        visit_count: Number of times the UAV has visited this node.
    """

    node_id: int
    position_ned: np.ndarray        # (3,)
    landmarks: list[Landmark] = field(default_factory=list)
    place_descriptor: Optional[np.ndarray] = None  # (D,)
    timestamp: float = 0.0
    keyframe_id: str = ""
    visit_count: int = 0

    @property
    def n_landmarks(self) -> int:
        """Number of semantic landmarks at this node."""
        return len(self.landmarks)

    def landmark_classes(self) -> set[str]:
        """Return the set of unique semantic class names present.

        Returns:
            Set of class name strings.
        """
        return {lm.semantic_class for lm in self.landmarks}


@dataclass
class TILMEdge:
    """A directed edge between two TILM nodes.

    Attributes:
        src_id: Source node identifier.
        dst_id: Destination node identifier.
        relative_pose: Relative SE(3) transform (4x4), float64.
            Transforms points in src frame to dst frame.
        distance: Euclidean distance between nodes in metres.
        traversal_count: Number of times this edge has been traversed.
        weight: Edge cost used for graph search (lower is better).
    """

    src_id: int
    dst_id: int
    relative_pose: np.ndarray   # (4, 4)
    distance: float
    traversal_count: int = 0
    weight: float = 1.0


class TILM:
    """Topological Invariant Landmark Map.

    A NetworkX-backed topological graph providing spatial indexing,
    nearest-node queries, path queries, and serialisation.

    Args:
        max_nodes: Maximum number of nodes before compaction. None = unlimited.
        min_node_distance: Minimum NED distance (m) to create a new node
            rather than merging with an existing nearby node.
    """

    def __init__(
        self,
        max_nodes: Optional[int] = None,
        min_node_distance: float = 5.0,
    ) -> None:
        self.max_nodes = max_nodes
        self.min_node_distance = min_node_distance
        self._graph: Optional[object] = None  # networkx.DiGraph
        self._node_positions: np.ndarray = np.empty((0, 3))

    def initialise(self) -> None:
        """Create the internal networkx graph.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.initialise is not yet implemented.")

    def add_node(self, node: TILMNode) -> int:
        """Insert a node into the graph, returning the assigned node ID.

        Args:
            node: TILMNode to insert.

        Returns:
            Assigned integer node ID.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.add_node is not yet implemented.")

    def add_edge(self, edge: TILMEdge) -> None:
        """Insert a directed edge between two existing nodes.

        Args:
            edge: TILMEdge connecting src_id → dst_id.

        Raises:
            NotImplementedError: Not yet implemented.
            KeyError: If either node ID does not exist.
        """
        raise NotImplementedError("TILM.add_edge is not yet implemented.")

    def nearest_node(
        self, position_ned: np.ndarray, k: int = 1
    ) -> list[tuple[int, float]]:
        """Find the k nearest nodes to a NED position.

        Args:
            position_ned: Query position, shape (3,), float64.
            k: Number of nearest nodes to return.

        Returns:
            List of (node_id, distance_m) tuples, sorted by distance ascending.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.nearest_node is not yet implemented.")

    def shortest_path(
        self, src_id: int, dst_id: int
    ) -> Optional[list[int]]:
        """Compute the shortest path between two nodes.

        Args:
            src_id: Source node ID.
            dst_id: Destination node ID.

        Returns:
            Ordered list of node IDs forming the path, or None if unreachable.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.shortest_path is not yet implemented.")

    def get_node(self, node_id: int) -> TILMNode:
        """Retrieve a node by ID.

        Args:
            node_id: Integer node identifier.

        Returns:
            The corresponding TILMNode.

        Raises:
            KeyError: If the node ID does not exist.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.get_node is not yet implemented.")

    def n_nodes(self) -> int:
        """Return the total number of nodes.

        Returns:
            Integer node count.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.n_nodes is not yet implemented.")

    def save(self, path: Path) -> None:
        """Serialise the map to a file (HDF5 or pickle).

        Args:
            path: Output file path.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.save is not yet implemented.")

    @classmethod
    def load(cls, path: Path) -> "TILM":
        """Load a TILM from a serialised file.

        Args:
            path: Path to a file produced by ``save()``.

        Returns:
            Loaded TILM instance.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TILM.load is not yet implemented.")
