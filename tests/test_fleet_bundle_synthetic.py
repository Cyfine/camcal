from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camcal import se3
from camcal.aruco import square_object_points
from camcal.fleet import (
    CameraFleetRecord,
    FleetCamera,
    FleetExtrinsics,
    FleetIntrinsics,
    load_fleet,
    save_fleet,
)
from camcal.fleet_bundle import (
    RoomStructure,
    derive_table_frame,
    initialize_fleet_poses,
    save_calibrated_fleet,
    solve_fleet_bundle,
)
from camcal.marker_map import MarkerManifest, MarkerSpec
from camcal.observations import CameraSnapshot, MarkerObservation, ObservationDataset


def _pose(rotvec_deg, translation) -> np.ndarray:
    return se3.from_Rt(
        Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix(),
        np.asarray(translation, dtype=np.float64),
    )


def _camera_pose(translation, tilt_deg) -> np.ndarray:
    # OpenCV optical +z points down toward the synthetic marker plane.
    rotation = np.diag([1.0, -1.0, -1.0]) @ Rotation.from_rotvec(
        np.deg2rad(tilt_deg)
    ).as_matrix()
    return se3.from_Rt(rotation, np.asarray(translation))


def _project(K, d, W_T_C, W_T_M, side_length_m) -> np.ndarray:
    C_T_M = se3.inverse(W_T_C) @ W_T_M
    rvec, tvec = se3.to_rvec_tvec(C_T_M)
    image, _ = cv2.projectPoints(
        square_object_points(side_length_m).reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        d,
    )
    return image.reshape(4, 2)


def _synthetic_dataset(noise_px: float):
    K = np.array([[950.0, 0, 640.0], [0, 960.0, 360.0], [0, 0, 1.0]])
    d = np.zeros(5)
    cameras_true = {
        "a": _camera_pose((-0.28, -0.10, 1.25), (3.0, -4.0, 1.0)),
        "b": _camera_pose((0.30, 0.00, 1.15), (-4.0, 3.0, -2.0)),
        "c": _camera_pose((0.02, 0.30, 1.30), (5.0, 2.0, 3.0)),
    }
    markers_true = {
        0: se3.identity(),
        1: _pose((0.0, 12.0, 8.0), (0.26, 0.10, 0.05)),
        2: _pose((-10.0, 0.0, -12.0), (-0.22, 0.22, 0.07)),
        3: _pose((7.0, -8.0, 18.0), (0.08, -0.26, 0.04)),
    }
    sizes = {0: 0.24, 1: 0.20, 2: 0.18, 3: 0.16}
    manifest = MarkerManifest(
        "DICT_6X6_250",
        tuple(
            MarkerSpec(marker_id, size, measured_size_m=size)
            for marker_id, size in sizes.items()
        ),
    )
    snapshots = tuple(
        CameraSnapshot(name, K, d, (1280, 720), intrinsics_ref=f"{name}.yaml")
        for name in sorted(cameras_true)
    )
    edges = {
        "a": (0, 1, 3),
        "b": (0, 2, 3),
        "c": (1, 2, 3),
    }
    rng = np.random.default_rng(4)
    observations = []
    for camera_name, marker_ids in edges.items():
        for marker_id in marker_ids:
            base = _project(
                K,
                d,
                cameras_true[camera_name],
                markers_true[marker_id],
                sizes[marker_id],
            )
            for sample in range(3):
                observations.append(
                    MarkerObservation(
                        camera_name=camera_name,
                        capture_id=f"{camera_name}-{marker_id}-{sample}",
                        marker_id=marker_id,
                        corners_px=base + rng.normal(0.0, noise_px, base.shape),
                    )
                )
    dataset = ObservationDataset(
        "DICT_6X6_250", snapshots, tuple(observations)
    )
    return dataset, manifest, cameras_true, markers_true


def test_noise_free_graph_initialization_has_correct_transform_direction():
    dataset, manifest, cameras_true, markers_true = _synthetic_dataset(0.0)

    cameras, markers = initialize_fleet_poses(
        dataset, manifest, anchor_marker_id=0
    )

    np.testing.assert_allclose(markers[0], np.eye(4), atol=1e-12)
    for name, truth in cameras_true.items():
        assert se3.translation_distance(cameras[name], truth) < 1e-4
        assert se3.rotation_distance(cameras[name], truth) < 0.01
    for marker_id, truth in markers_true.items():
        assert se3.translation_distance(markers[marker_id], truth) < 1e-4
        assert se3.rotation_distance(markers[marker_id], truth) < 0.01


def test_joint_bundle_recovers_multi_camera_multi_marker_scene_with_noise():
    dataset, manifest, cameras_true, markers_true = _synthetic_dataset(0.20)

    result = solve_fleet_bundle(
        dataset,
        manifest,
        anchor_marker_id=0,
        loss="soft_l1",
        f_scale_px=1.0,
        max_nfev=300,
    )

    assert result.success, result.message
    assert result.final_raw_cost < result.initial_raw_cost
    assert result.global_residual.rmse_px < 0.35
    np.testing.assert_allclose(result.marker_poses_W_T_M[0], np.eye(4), atol=1e-12)
    for name, truth in cameras_true.items():
        assert se3.translation_distance(result.camera_poses_W_T_C[name], truth) < 0.006
        assert se3.rotation_distance(result.camera_poses_W_T_C[name], truth) < 0.4
    for marker_id, truth in markers_true.items():
        assert se3.translation_distance(result.marker_poses_W_T_M[marker_id], truth) < 0.006
        assert se3.rotation_distance(result.marker_poses_W_T_M[marker_id], truth) < 0.4


def test_nearly_frontal_repeated_square_observations_do_not_mix_ippe_branches():
    K = np.array([[1100.0, 0, 640.0], [0, 1100.0, 360.0], [0, 0, 1.0]])
    d = np.zeros(5)
    camera_true = _camera_pose((0.01, -0.02, 2.8), (0.18, -0.22, 0.05))
    side = 0.24
    manifest = MarkerManifest(
        "DICT_6X6_250",
        (MarkerSpec(0, side, measured_size_m=side),),
    )
    rng = np.random.default_rng(12)
    base = _project(K, d, camera_true, np.eye(4), side)
    dataset = ObservationDataset(
        "DICT_6X6_250",
        (CameraSnapshot("ceiling", K, d, (1280, 720)),),
        tuple(
            MarkerObservation(
                "ceiling",
                f"frame-{index}",
                0,
                base + rng.normal(0.0, 0.08, base.shape),
            )
            for index in range(10)
        ),
    )

    result = solve_fleet_bundle(
        dataset, manifest, anchor_marker_id=0, f_scale_px=0.5
    )

    assert result.success, result.message
    assert se3.translation_distance(
        result.camera_poses_W_T_C["ceiling"], camera_true
    ) < 0.025
    assert se3.rotation_distance(
        result.camera_poses_W_T_C["ceiling"], camera_true
    ) < 1.0


def test_calibrated_fleet_uses_anchor_world_and_clears_unsolved_stale_pose(tmp_path):
    dataset, manifest, _, _ = _synthetic_dataset(0.10)
    result = solve_fleet_bundle(dataset, manifest, anchor_marker_id=0)
    K = dataset.cameras[0].K
    d = dataset.cameras[0].d
    fleet = CameraFleetRecord(
        name="mixed-before-save",
        world_frame="old_world",
        cameras=[
            *[
                FleetCamera(
                    name=name,
                    intrinsics=FleetIntrinsics(
                        source="test", image_size=(1280, 720), K=K, d=d
                    ),
                )
                for name in ("a", "b", "c")
            ],
            FleetCamera(
                name="not-observed",
                extrinsics=FleetExtrinsics(W_T_C=np.eye(4)),
            ),
        ],
    )
    source = tmp_path / "source-fleet.yaml"
    output = tmp_path / "solved-fleet.yaml"
    save_fleet(source, fleet)

    save_calibrated_fleet(source, output, result)

    loaded = load_fleet(output)
    assert loaded.world_frame == "aruco_marker_0"
    assert loaded.get_camera("not-observed").extrinsics is None
    assert loaded.get_camera("a").extrinsics is not None


def test_table_frame_is_composed_from_room_marker_pose_in_correct_direction():
    dataset, manifest, cameras_true, markers_true = _synthetic_dataset(0.0)
    result = solve_fleet_bundle(dataset, manifest, anchor_marker_id=0)
    table_T_marker = _pose((12.0, -7.0, 5.0), (0.11, -0.09, 0.75))

    table_frame = derive_table_frame(
        result,
        table_marker_id=3,
        table_T_marker=table_T_marker,
    )

    expected_room_T_table = markers_true[3] @ se3.inverse(table_T_marker)
    np.testing.assert_allclose(
        table_frame.room_T_table, expected_room_T_table, atol=1e-5
    )
    np.testing.assert_allclose(
        table_frame.table_T_room @ table_frame.room_T_table,
        np.eye(4),
        atol=1e-12,
    )
    expected_table_T_camera_a = se3.inverse(expected_room_T_table) @ cameras_true["a"]
    actual_table_T_camera_a = (
        table_frame.table_T_room @ result.camera_poses_W_T_C["a"]
    )
    np.testing.assert_allclose(
        actual_table_T_camera_a, expected_table_T_camera_a, atol=1e-5
    )


def test_structured_parameterization_enforces_room_planes_exactly():
    dataset, manifest, _, _ = _synthetic_dataset(0.10)
    structure = RoomStructure(
        floor_marker_id=0,
        parallel_wall_marker_groups=((1, 3), (2,)),
    )

    result = solve_fleet_bundle(
        dataset,
        manifest,
        anchor_marker_id=0,
        room_structure=structure,
        max_nfev=300,
    )

    assert result.success, result.message
    diagnostics = result.structure_diagnostics
    assert diagnostics is not None
    assert max(diagnostics["wall_floor_perpendicular_error_deg"].values()) < 1e-6
    assert max(diagnostics["wall_parallel_error_deg"].values()) < 1e-6
    assert max(diagnostics["same_wall_center_plane_error_m"].values()) < 1e-10
