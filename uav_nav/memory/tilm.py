"""Topological Invariant Landmark Map (TILM).

The TILM is a topological graph where each node represents a geographic
location annotated with a set of semantic landmarks observed from that
position. Edges encode traversability and relative pose constraints.

The map is weather-invariant because it uses semantic and geometric
descriptors rather than raw appearance features.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
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
    position_ned: np.ndarray              # (3,)
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
        """Return the set of unique semantic class names present."""
        return {lm.semantic_class for lm in self.landmarks}


@dataclass
class TILMEdge:
    """A directed edge between two TILM nodes.

    Attributes:
        src_id: Source node identifier.
        dst_id: Destination node identifier.
        relative_pose: Relative SE(3) transform (4×4), float64.
        distance: Euclidean distance between nodes in metres.
        traversal_count: Number of times this edge has been traversed.
        weight: Edge cost used for graph search (lower is better).
    """

    src_id: int
    dst_id: int
    relative_pose: np.ndarray    # (4, 4)
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
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[int, TILMNode] = {}
        self._node_ids: list[int] = []
        self._positions: list[np.ndarray] = []

    def initialise(self) -> None:
        """Reset the internal graph to an empty state."""
        self._graph = nx.DiGraph()
        self._nodes = {}
        self._node_ids = []
        self._positions = []

    def add_node(self, node: TILMNode) -> int:
        """Insert a node into the graph.

        Args:
            node: TILMNode to insert.

        Returns:
            Assigned integer node ID.
        """
        nid = node.node_id
        self._graph.add_node(nid)
        self._nodes[nid] = node
        self._node_ids.append(nid)
        self._positions.append(node.position_ned.copy())
        return nid

    def add_edge(self, edge: TILMEdge) -> None:
        """Insert a directed edge between two existing nodes.

        Args:
            edge: TILMEdge connecting src_id → dst_id.

        Raises:
            KeyError: If either node ID does not exist.
        """
        for nid in (edge.src_id, edge.dst_id):
            if nid not in self._nodes:
                raise KeyError(f"Node {nid} not in TILM")
        self._graph.add_edge(edge.src_id, edge.dst_id, data=edge)

    def nearest_node(
        self, position_ned: np.ndarray, k: int = 1
    ) -> list[tuple[int, float]]:
        """Find the k nearest nodes to a NED position.

        Args:
            position_ned: Query position, shape (3,), float64.
            k: Number of nearest nodes to return.

        Returns:
            List of (node_id, distance_m) tuples, sorted ascending by distance.
        """
        if not self._node_ids:
            return []
        pos_arr = np.array(self._positions)              # (N, 3)
        dists = np.linalg.norm(pos_arr - position_ned, axis=1)
        k = min(k, len(self._node_ids))
        top_idx = np.argpartition(dists, k - 1)[:k]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return [(self._node_ids[i], float(dists[i])) for i in top_idx]

    def shortest_path(
        self, src_id: int, dst_id: int
    ) -> Optional[list[int]]:
        """Compute the shortest path between two nodes.

        Args:
            src_id: Source node ID.
            dst_id: Destination node ID.

        Returns:
            Ordered list of node IDs, or None if unreachable.
        """
        try:
            return nx.shortest_path(self._graph, src_id, dst_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_node(self, node_id: int) -> TILMNode:
        """Retrieve a node by ID.

        Args:
            node_id: Integer node identifier.

        Returns:
            The corresponding TILMNode.

        Raises:
            KeyError: If the node ID does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not in TILM")
        return self._nodes[node_id]

    def n_nodes(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    @property
    def all_nodes(self) -> list[TILMNode]:
        """Return all nodes as a list."""
        return list(self._nodes.values())

    def save(self, path: Path) -> None:
        """Serialise the map to a pickle file.

        Args:
            path: Output file path (.tilm or .pkl).
        """
        path = Path(path)
        data = {
            "max_nodes": self.max_nodes,
            "min_node_distance": self.min_node_distance,
            "graph": self._graph,
            "nodes": self._nodes,
            "node_ids": self._node_ids,
            "positions": self._positions,
        }
        path.write_bytes(pickle.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "TILM":
        """Load a TILM from a serialised file.

        Args:
            path: Path to a file produced by ``save()``.

        Returns:
            Loaded TILM instance.
        """
        data = pickle.loads(Path(path).read_bytes())
        tilm = cls(
            max_nodes=data["max_nodes"],
            min_node_distance=data["min_node_distance"],
        )
        tilm._graph = data["graph"]
        tilm._nodes = data["nodes"]
        tilm._node_ids = data["node_ids"]
        tilm._positions = data["positions"]
        return tilm
