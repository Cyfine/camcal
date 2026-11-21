"""Moving-board bundle adjustment on synthetic three-camera data."""

from __future__ import annotations

import numpy as np
import yaml
import pytest
from scipy.spatial.transform import Rotation

from camcal import se3
from camcal.board_bundle import (
    W_T_B_WORLD,
    DroppedView,
    _project,
    image_coverage,
    diagnose_rig,
    intrinsics_displacement_px,
    solve_board_bundle,
    validate_transfer,
)
from camcal.board_observations import (
    BoardCapture,
    BoardObservation,
    BoardObservationDataset,
    board_corner_points,
    load_board_observations,
    save_board_observations,
)
from camcal.charuco import CharucoConfig
from camcal.observations import CameraSnapshot

BOARD = CharucoConfig(6, 6, 0.12, 0.09, "DICT_4X4_50", legacy_pattern=True)
SIZE = (1920, 1080)
TRUE_K = {
    "a": np.array([[1020.0, 0, 990.0], [0, 1019.0, 610.0], [0, 0, 1]]),
    "c": np.array([[1015.0, 0, 860.0], [0, 1016.0, 570.0], [0, 0, 1]]),
    "d415": np.array([[1380.0, 0, 955.0], [0, 1380.0, 540.0], [0, 0, 1]]),
}
TRUE_D = {
    "a": np.array([0.13, -0.24, 0.0, 0.0, 0.10]),
    "c": np.array([0.14, -0.27, 0.0, 0.0, 0.13]),
    "d415": np.zeros(5),
}


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return se3.from_Rt(np.column_stack((x, y, z)), np.asarray(position, float))


TRUE_POSES = {
    "a": look_at([-1.4, -2.6, 2.0], [0.0, 0.0, 0.6]),
    "c": look_at([1.3, -2.7, 2.1], [0.0, 0.0, 0.6]),
    "d415": look_at([0.1, -1.6, 1.0], [0.0, 0.0, 0.6]),
}


def board_poses(count, rng):
    """Every third pose is in the shared volume; the others sweep one camera's image."""
    poses = []
    names = list(TRUE_POSES)
    while len(poses) < count:
        tilt = Rotation.from_euler("xyz", rng.uniform(-35, 35, 3), degrees=True)
        if len(poses) % 3 == 0:
            center = np.array([0.0, 0.0, 0.7]) + rng.uniform([-0.6, -0.5, -0.4], [0.6, 0.5, 0.5])
            R = (Rotation.from_euler("x", -90, degrees=True) * tilt).as_matrix()  # faces -y
        else:
            name = names[len(poses) % len(names)]
            near, far = (0.9, 1.8) if name == "d415" else (1.3, 2.6)
            pixel = np.r_[rng.uniform(100, SIZE[0] - 100), rng.uniform(100, SIZE[1] - 100), 1.0]
            ray = np.linalg.solve(TRUE_K[name], pixel)
            center_c = ray / ray[2] * rng.uniform(near, far)
            W_T_C = TRUE_POSES[name]
            center = W_T_C[:3, :3] @ center_c + W_T_C[:3, 3]
            R = W_T_C[:3, :3] @ tilt.as_matrix()  # faces that camera
        poses.append(se3.from_Rt(R, center - R @ np.array([0.36, 0.36, 0.0])))
    return poses


def observe(W_T_B, name, rng, noise_px):
    corners = board_corner_points(BOARD)
    C_T_B = se3.compose(se3.inverse(TRUE_POSES[name]), W_T_B)
    points_c = corners @ C_T_B[:3, :3].T + C_T_B[:3, 3]
    out_normal_c = -C_T_B[:3, 2]
    if np.any(points_c[:, 2] < 0.3) or np.dot(out_normal_c, -points_c.mean(axis=0)) <= 0:
        return None
    intr = np.r_[TRUE_K[name][0, 0], TRUE_K[name][1, 1], TRUE_K[name][0, 2], TRUE_K[name][1, 2], TRUE_D[name]]
    pixels = _project(points_c, intr) + rng.normal(0.0, noise_px, (len(corners), 2))
    inside = (pixels[:, 0] > 5) & (pixels[:, 0] < SIZE[0] - 5) & (pixels[:, 1] > 5) & (pixels[:, 1] < SIZE[1] - 5)
    if inside.sum() < 8:
        return None
    return np.flatnonzero(inside), pixels[inside]


def make_dataset(rng, *, solve=40, validation=6, noise_px=0.2, prior_scale=1.02, poses=None):
    poses = dict(TRUE_POSES if poses is None else poses)
    captures, observations = [], []
    W_T_B = [W_T_B_WORLD] + board_poses(solve + validation, rng)
    for index, pose in enumerate(W_T_B):
        role = "world" if index == 0 else ("validation" if index > solve else "solve")
        capture_id = f"cap-{index:03d}"
        seen = 0
        for name in TRUE_POSES:
            hit = observe(pose, name, rng, noise_px)
            if hit is not None:
                ids, pixels = hit
                observations.append(BoardObservation(name, capture_id, ids, pixels))
                seen += 1
        if seen:
            captures.append(BoardCapture(capture_id, role))
    cameras = []
    for name in TRUE_POSES:
        K = TRUE_K[name].copy()
        if name != "d415":
            K[:2, :2] *= prior_scale  # the focus ring moved
        cameras.append(CameraSnapshot(name, K, TRUE_D[name] if name == "d415" else np.zeros(5), SIZE))
    return BoardObservationDataset(BOARD, tuple(cameras), tuple(captures), tuple(observations)), poses


def test_full_calibration_recovers_poses_and_intrinsics():
    rng = np.random.default_rng(3)
    dataset, _ = make_dataset(rng)
    # The world capture lies on the floor; it is seen by the high cameras.
    assert any(obs.capture_id == "cap-000" for obs in dataset.observations)
    result = solve_board_bundle(dataset, free_intrinsics=["a", "c"])
    assert result.success
    names = list(TRUE_POSES)
    for name, truth in TRUE_POSES.items():
        # Absolute errors include how precisely the one world capture fixes the world.
        assert se3.translation_distance(result.W_T_C[name], truth) < 0.006, name
        assert se3.rotation_distance(result.W_T_C[name], truth) < 0.15, name
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            solved = se3.compose(se3.inverse(result.W_T_C[first]), result.W_T_C[second])
            truth = se3.compose(se3.inverse(TRUE_POSES[first]), TRUE_POSES[second])
            assert se3.translation_distance(solved, truth) < 0.003, (first, second)
    for name in ("a", "c"):
        assert np.allclose(result.K[name], TRUE_K[name], rtol=2e-3, atol=1.5)
        seen = np.concatenate([obs.image_points for obs in dataset.observations if obs.camera_name == name])
        assert image_coverage(seen, SIZE) > 0.3
        shift = intrinsics_displacement_px(
            TRUE_K[name], TRUE_D[name], result.K[name], result.d[name], SIZE, observed_points=seen
        )
        assert shift["max_px"] < 1.5, (name, shift)
    assert result.camera_stats["a"].rmse_px < 0.4
    # The D415 keeps its prior intrinsics.
    assert np.allclose(result.K["d415"], TRUE_K["d415"])

    rows = validate_transfer(dataset, result.K, result.d, result.W_T_C)
    assert rows
    assert max(row["rmse_px"] for row in rows) < 1.5


def test_world_capture_gives_z_up_world():
    rng = np.random.default_rng(5)
    dataset, _ = make_dataset(rng, solve=30, validation=0)
    result = solve_board_bundle(dataset, free_intrinsics=["a", "c"])
    # Cameras above the floor board have positive z in the world.
    for name in TRUE_POSES:
        assert result.W_T_C[name][2, 3] > 0.5


def test_bumped_camera_is_found_and_recalibrated():
    rng = np.random.default_rng(11)
    bumped = dict(TRUE_POSES)
    bump = se3.from_Rt(Rotation.from_euler("z", 1.5, degrees=True).as_matrix(), np.array([0.02, 0.0, 0.0]))
    dataset, _ = make_dataset(rng, solve=12, validation=0, prior_scale=1.0)
    stale = dict(TRUE_POSES)
    stale["c"] = se3.compose(TRUE_POSES["c"], se3.inverse(bump))  # the rig still holds the old pose
    snapshots = tuple(
        snapshot if snapshot.name == "d415" else type(snapshot)(snapshot.name, TRUE_K[snapshot.name], TRUE_D[snapshot.name], SIZE)
        for snapshot in dataset.cameras
    )
    dataset = BoardObservationDataset(BOARD, snapshots, dataset.captures, dataset.observations)

    report = diagnose_rig(dataset, stale)
    assert report["baseline_rmse_px"] > 2.0
    freed = report["freed"]
    assert min(freed, key=lambda name: freed[name]["rmse_px"]) == "c"
    assert freed["c"]["rmse_px"] < 0.5
    assert freed["a"]["rmse_px"] > 1.0
    assert freed["c"]["translation_change_mm"] > 10

    fixed = {name: stale[name] for name in ("a", "d415")}
    result = solve_board_bundle(dataset, fixed_poses=fixed, initial_poses={"c": stale["c"]})
    assert se3.translation_distance(result.W_T_C["c"], TRUE_POSES["c"]) < 0.003
    assert np.allclose(result.W_T_C["a"], TRUE_POSES["a"])


def test_dataset_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    dataset, _ = make_dataset(rng, solve=4, validation=1)
    path = tmp_path / "observations.yaml"
    save_board_observations(path, dataset)
    loaded = load_board_observations(path)
    assert loaded.board == BOARD
    assert loaded.world_capture == "cap-000"
    assert [c.capture_id for c in loaded.captures] == [c.capture_id for c in dataset.captures]
    for a, b in zip(loaded.observations, dataset.observations):
        assert np.array_equal(a.corner_ids, b.corner_ids)
        assert np.allclose(a.image_points, b.image_points)


def test_full_calibration_needs_a_world_capture():
    rng = np.random.default_rng(4)
    dataset, _ = make_dataset(rng, solve=8, validation=0)
    captures = tuple(BoardCapture(c.capture_id, "solve") for c in dataset.captures)
    dataset = BoardObservationDataset(BOARD, dataset.cameras, captures, dataset.observations)
    with pytest.raises(ValueError, match="world capture"):
        solve_board_bundle(dataset)


def test_legacy_board_corners_match_opencv_detection_layout():
    corners = board_corner_points(BOARD)
    assert corners.shape == (25, 3)
    assert np.allclose(corners[0], [0.12, 0.12, 0.0])
    assert np.allclose(corners[24], [0.60, 0.60, 0.0])


def _fleet(path, poses):
    from camcal.fleet import CameraFleetRecord, FleetCamera, FleetExtrinsics, FleetIntrinsics, save_fleet

    cameras = [
        FleetCamera(
            name,
            intrinsics=FleetIntrinsics(source="prior", image_size=SIZE, K=TRUE_K[name], d=TRUE_D[name]),
            extrinsics=FleetExtrinsics(detection_mode="old", W_T_C=poses[name]),
        )
        for name in TRUE_POSES
    ]
    save_fleet(path, CameraFleetRecord("synthetic", cameras, world_frame="old_world"))


def test_cli_full_check_and_bumped(tmp_path):
    from camcal.calibrate_board_fleet import main
    from camcal.fleet import load_fleet

    rng = np.random.default_rng(3)
    dataset, _ = make_dataset(rng, solve=30, validation=3)
    observations = tmp_path / "observations.yaml"
    save_board_observations(observations, dataset)
    fleet = tmp_path / "fleet.yaml"
    _fleet(fleet, TRUE_POSES)

    out_fleet = tmp_path / "calibrated.yaml"
    assert main([
        "--fleet", str(fleet), "--observations", str(observations), "--out", str(tmp_path / "full.yaml"),
        "--out-fleet", str(out_fleet), "--free-intrinsics", "a", "--free-intrinsics", "c",
    ]) == 0
    calibrated = load_fleet(out_fleet)
    assert calibrated.world_frame == "board_world"
    a = calibrated.get_camera("a")
    assert a.extrinsics.detection_mode == "charuco_moving_board"
    assert a.intrinsics.source.startswith("charuco_moving_board")
    assert calibrated.get_camera("d415").intrinsics.source == "prior"
    report = yaml.safe_load((tmp_path / "full.yaml").read_text())
    assert report["validation"]

    assert main([
        "--fleet", str(out_fleet), "--observations", str(observations), "--out", str(tmp_path / "check.yaml"), "--check",
    ]) == 0
    check = yaml.safe_load((tmp_path / "check.yaml").read_text())
    assert check["diagnosis"]["baseline_rmse_px"] < 1.0

    bumped_fleet = tmp_path / "bumped.yaml"
    assert main([
        "--fleet", str(out_fleet), "--observations", str(observations), "--out", str(tmp_path / "bumped-report.yaml"),
        "--out-fleet", str(bumped_fleet), "--fix-pose", "a", "--fix-pose", "d415",
    ]) == 0
    bumped = load_fleet(bumped_fleet)
    assert bumped.world_frame == "board_world"
    assert np.allclose(bumped.get_camera("a").extrinsics.W_T_C, a.extrinsics.W_T_C)


def test_auto_capture_takes_only_new_views():
    from camcal.charuco import CharucoDetection
    from camcal.observe_board import AutoSettings, View, corner_motion_px, is_new_view

    settings = AutoSettings()
    first = View(np.array([960.0, 540.0]), np.eye(3), 2.0)
    same = View(np.array([970.0, 545.0]), Rotation.from_euler("x", 3, degrees=True).as_matrix(), 2.05)
    shifted = View(np.array([1400.0, 540.0]), np.eye(3), 2.0)
    tilted = View(np.array([960.0, 540.0]), Rotation.from_euler("y", 25, degrees=True).as_matrix(), 2.0)
    closer = View(np.array([960.0, 540.0]), np.eye(3), 1.4)
    assert is_new_view(first, [], 1920, settings)
    assert not is_new_view(same, [first], 1920, settings)
    for view in (shifted, tilted, closer):
        assert is_new_view(view, [first], 1920, settings)
    assert not is_new_view(shifted, [first, shifted], 1920, settings)

    ids = np.arange(10)
    points = np.column_stack((np.arange(10.0) * 50, np.zeros(10)))
    still = CharucoDetection(np.zeros((10, 3)), points + 0.2, ids)
    moved = CharucoDetection(np.zeros((10, 3)), points + 5.0, ids)
    base = CharucoDetection(np.zeros((10, 3)), points, ids)
    # One loop iteration of about 0.1 s.
    assert corner_motion_px(base, still) < settings.still_px_per_s * 0.1
    assert corner_motion_px(base, moved) > settings.still_px_per_s * 0.1
    assert corner_motion_px(None, still) == float("inf")



def _with_one_row_view(dataset, name, capture_id, W_T_B, rng):
    """Replace ``name``'s view of ``capture_id`` by one board row and one more corner.

    Five of the six corners lie on one line, so they do not fix the board's pose.
    """
    C_T_B = se3.compose(se3.inverse(TRUE_POSES[name]), W_T_B)
    corners = board_corner_points(BOARD)
    ids = np.arange(6)
    points_c = corners[ids] @ C_T_B[:3, :3].T + C_T_B[:3, 3]
    intr = np.r_[TRUE_K[name][0, 0], TRUE_K[name][1, 1], TRUE_K[name][0, 2], TRUE_K[name][1, 2], TRUE_D[name]]
    pixels = _project(points_c, intr) + rng.normal(0.0, 0.2, (len(ids), 2))
    observations = [obs for obs in dataset.observations if (obs.camera_name, obs.capture_id) != (name, capture_id)]
    observations.append(BoardObservation(name, capture_id, ids, pixels))
    return BoardObservationDataset(dataset.board, dataset.cameras, dataset.captures, tuple(observations))


def test_view_whose_corners_do_not_fix_the_board_is_left_out():
    rng = np.random.default_rng(3)
    poses = [W_T_B_WORLD] + board_poses(46, np.random.default_rng(3))
    dataset, _ = make_dataset(np.random.default_rng(3))
    # cap-003 is in the shared volume (every third pose) and seen by every camera.
    shared = "cap-003"
    assert {obs.camera_name for obs in dataset.observations if obs.capture_id == shared} == set(TRUE_POSES)
    dataset = _with_one_row_view(dataset, "d415", shared, poses[3], rng)
    # In a validation capture such a view comes first, where the transfer check seeds the board pose.
    validation = next(c.capture_id for c in dataset.captures if c.role == "validation")
    dataset = _with_one_row_view(dataset, "a", validation, poses[int(validation[4:])], rng)
    dataset = BoardObservationDataset(
        dataset.board,
        dataset.cameras,
        dataset.captures,
        tuple(sorted(dataset.observations, key=lambda obs: (obs.capture_id, obs.camera_name))),
    )

    result = solve_board_bundle(dataset, free_intrinsics=["a", "c"])
    assert result.dropped_views == (DroppedView("d415", shared, 6),)
    assert ("d415", shared) not in result.observation_rmse_px
    for name, truth in TRUE_POSES.items():
        assert se3.translation_distance(result.W_T_C[name], truth) < 0.006, name
    assert result.camera_stats["a"].rmse_px < 0.4

    rows = validate_transfer(dataset, result.K, result.d, result.W_T_C)
    assert any(row["capture"] == validation for row in rows)


def test_camera_whose_only_views_are_left_out_is_not_linked():
    dataset, _ = make_dataset(np.random.default_rng(3), solve=12, validation=0)
    poses = [W_T_B_WORLD] + board_poses(12, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    for obs in [obs for obs in dataset.observations if obs.camera_name == "d415"]:
        dataset = _with_one_row_view(dataset, "d415", obs.capture_id, poses[int(obs.capture_id[4:])], rng)
    with pytest.raises(ValueError, match="not linked"):
        solve_board_bundle(dataset, free_intrinsics=["a", "c"])


def test_huber_solve_converges_from_a_start_with_most_corners_past_the_knee():
    # Cameras 1-3 cm off: most corners start well past the 2 px knee. scipy's own Huber model
    # gives those corners no curvature, and the solve used to stop near its start at ~6 px.
    dataset, _ = make_dataset(np.random.default_rng(3), solve=30, validation=0)
    start = {}
    for k, (name, truth) in enumerate(TRUE_POSES.items(), start=1):
        error = se3.from_Rt(
            Rotation.from_euler("xyz", [0.3, -0.3, 0.3 * k], degrees=True).as_matrix(), np.array([0.01, -0.01, 0.01 * k])
        )
        start[name] = se3.compose(truth, error)
    result = solve_board_bundle(dataset, free_intrinsics=["a", "c"], initial_poses=start, loss="huber", f_scale_px=2.0)
    for name, truth in TRUE_POSES.items():
        assert se3.translation_distance(result.W_T_C[name], truth) < 0.006, name
    assert max(stats.rmse_px for stats in result.camera_stats.values()) < 0.4
