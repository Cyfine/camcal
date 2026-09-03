"""Camera--marker bipartite graph checks for fleet calibration."""

from __future__ import annotations

from dataclasses import dataclass

from .observations import ObservationDataset


@dataclass(frozen=True)
class GraphComponent:
    cameras: tuple[str, ...]
    markers: tuple[int, ...]

    def describe(self) -> str:
        return f"cameras={list(self.cameras)}, markers={list(self.markers)}"


@dataclass(frozen=True)
class ObservationGraph:
    components: tuple[GraphComponent, ...]
    edge_counts: dict[tuple[str, int], int]

    @property
    def connected(self) -> bool:
        return len(self.components) == 1


class DisconnectedObservationGraph(ValueError):
    """The observations do not determine one common camera/marker frame."""

    def __init__(self, graph: ObservationGraph) -> None:
        details = "; ".join(
            f"component {index + 1}: {component.describe()}"
            for index, component in enumerate(graph.components)
        )
        super().__init__(
            "camera-marker observation graph is disconnected; no single common "
            f"frame exists. {details}"
        )
        self.graph = graph


def build_observation_graph(dataset: ObservationDataset) -> ObservationGraph:
    """Build components over every dataset camera and every observed marker."""
    camera_nodes = {("camera", camera.name) for camera in dataset.cameras}
    marker_ids = {observation.marker_id for observation in dataset.observations}
    marker_nodes = {("marker", marker_id) for marker_id in marker_ids}
    nodes: set[tuple[str, object]] = set(camera_nodes | marker_nodes)
    adjacency = {node: set() for node in nodes}
    edge_counts: dict[tuple[str, int], int] = {}
    for observation in dataset.observations:
        camera = ("camera", observation.camera_name)
        marker = ("marker", observation.marker_id)
        adjacency[camera].add(marker)
        adjacency[marker].add(camera)
        edge = (observation.camera_name, observation.marker_id)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    components: list[GraphComponent] = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen, key=lambda node: (node[0], str(node[1])))
        stack = [start]
        reached: set[tuple[str, object]] = set()
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency[node] - reached)
        unseen -= reached
        components.append(
            GraphComponent(
                cameras=tuple(
                    sorted(str(value) for kind, value in reached if kind == "camera")
                ),
                markers=tuple(
                    sorted(int(value) for kind, value in reached if kind == "marker")
                ),
            )
        )
    components.sort(key=lambda component: (component.cameras, component.markers))
    return ObservationGraph(tuple(components), edge_counts)


def require_connected_graph(
    dataset: ObservationDataset,
    *,
    anchor_marker_id: int,
) -> ObservationGraph:
    """Return the graph or raise a plain observability error."""
    graph = build_observation_graph(dataset)
    observed_markers = {
        marker for component in graph.components for marker in component.markers
    }
    if anchor_marker_id not in observed_markers:
        raise ValueError(
            f"anchor marker {anchor_marker_id} was not observed; it defines world"
        )
    if not graph.connected:
        raise DisconnectedObservationGraph(graph)
    return graph


__all__ = [
    "DisconnectedObservationGraph",
    "GraphComponent",
    "ObservationGraph",
    "build_observation_graph",
    "require_connected_graph",
]
