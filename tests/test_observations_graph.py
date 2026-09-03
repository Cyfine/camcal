from __future__ import annotations

import numpy as np
import pytest

from camcal.observation_graph import (
    DisconnectedObservationGraph,
    build_observation_graph,
    require_connected_graph,
)
from camcal.observations import (
    CameraSnapshot,
    MarkerObservation,
    ObservationDataset,
    load_observation_dataset,
    merge_observation_datasets,
    save_observation_dataset,
)


def _camera(name: str) -> CameraSnapshot:
    return CameraSnapshot(
        name=name,
        K=np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        d=np.zeros(5),
        image_size=(640, 480),
        intrinsics_ref=f"{name}.yaml",
    )


def _observation(
    camera: str,
    marker: int,
    *,
    source_image: str = "",
) -> MarkerObservation:
    return MarkerObservation(
        camera_name=camera,
        capture_id=f"{camera}-{marker}",
        marker_id=marker,
        corners_px=np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
        source_image=source_image,
        source_host="lab",
    )


def test_connected_graph_can_bridge_cameras_through_different_markers():
    dataset = ObservationDataset(
        dictionary="DICT_6X6_250",
        cameras=(_camera("a"), _camera("b"), _camera("c")),
        observations=(
            _observation("a", 0),
            _observation("a", 1),
            _observation("b", 1),
            _observation("b", 2),
            _observation("c", 2),
        ),
    )

    graph = require_connected_graph(dataset, anchor_marker_id=0)

    assert graph.connected
    assert graph.components[0].cameras == ("a", "b", "c")
    assert graph.components[0].markers == (0, 1, 2)


def test_disconnected_graph_names_each_independent_world():
    dataset = ObservationDataset(
        dictionary="DICT_6X6_250",
        cameras=(_camera("a"), _camera("b")),
        observations=(_observation("a", 0), _observation("b", 7)),
    )

    graph = build_observation_graph(dataset)

    assert not graph.connected
    with pytest.raises(DisconnectedObservationGraph, match="camera-marker.*disconnected") as caught:
        require_connected_graph(dataset, anchor_marker_id=0)
    assert "cameras=['a']" in str(caught.value)
    assert "cameras=['b']" in str(caught.value)


def test_observation_yaml_round_trip_and_cross_host_merge(tmp_path):
    first = ObservationDataset(
        dictionary="DICT_6X6_250",
        cameras=(_camera("a"),),
        observations=(_observation("a", 0, source_image="a_images/frame.png"),),
    )
    second = ObservationDataset(
        dictionary="DICT_6X6_250",
        cameras=(_camera("b"),),
        observations=(_observation("b", 0),),
    )
    path = tmp_path / "a.yaml"
    save_observation_dataset(path, first)

    loaded = load_observation_dataset(path)
    merged = merge_observation_datasets((loaded, second))

    assert [camera.name for camera in merged.cameras] == ["a", "b"]
    assert len(merged.observations) == 2
    assert loaded.observations[0].source_image == str(
        (tmp_path / "a_images/frame.png").resolve()
    )
    assert require_connected_graph(merged, anchor_marker_id=0).connected


def test_merge_rejects_same_camera_name_with_different_intrinsics():
    altered = _camera("a")
    altered = CameraSnapshot(
        name="a",
        K=altered.K + np.diag([1.0, 0.0, 0.0]),
        d=altered.d,
        image_size=altered.image_size,
    )
    first = ObservationDataset(
        "DICT_6X6_250", (_camera("a"),), (_observation("a", 0),)
    )
    second = ObservationDataset(
        "DICT_6X6_250", (altered,), (_observation("a", 1),)
    )

    with pytest.raises(ValueError, match="conflicting intrinsics"):
        merge_observation_datasets((first, second))
