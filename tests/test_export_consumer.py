from __future__ import annotations

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R_scipy

from camcal import se3
from camcal.charuco import CharucoConfig
from camcal.export_consumer import export_consumer_bundle
from camcal.fleet import (
    CameraFleetRecord,
    FleetCamera,
    FleetExtrinsics,
    FleetIntrinsics,
    save_fleet,
)
from camcal.io_yaml import (
    ExtrinsicsRecord,
    IntrinsicsRecord,
    save_extrinsics,
    save_intrinsics,
)


def _read_opencv_matrix(path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        return np.asarray(fs.getNode(key).mat(), dtype=np.float64)
    finally:
        fs.release()


def test_export_consumer_bundle_from_inline_fleet_state(tmp_path):
    K0 = np.array([[800.0, 0.0, 320.0], [0.0, 805.0, 240.0], [0.0, 0.0, 1.0]])
    d0 = np.array([0.01, -0.02, 0.001, 0.0, 0.0])
    K2 = np.array([[810.0, 0.0, 315.0], [0.0, 812.0, 245.0], [0.0, 0.0, 1.0]])
    d2 = np.array([-0.01, 0.02, 0.0, 0.001, 0.0])

    W_T_C0 = se3.identity()
    W_T_C2 = se3.from_Rt(
        R_scipy.from_euler("xyz", [3.0, -2.0, 5.0], degrees=True).as_matrix(),
        np.array([0.25, -0.03, 0.01]),
    )

    fleet = CameraFleetRecord(
        name="alpha-rig",
        cameras=[
            FleetCamera(
                name="front-left",
                device={"serial": "A0"},
                intrinsics=FleetIntrinsics(
                    source="zhang",
                    image_size=(640, 480),
                    K=K0,
                    d=d0,
                ),
                extrinsics=FleetExtrinsics(
                    W_T_C=W_T_C0,
                    detection_mode="charuco",
                ),
            ),
            FleetCamera(
                name="front-right",
                device={"serial": "A2"},
                intrinsics=FleetIntrinsics(
                    source="zhang",
                    image_size=(640, 480),
                    K=K2,
                    d=d2,
                ),
                extrinsics=FleetExtrinsics(
                    W_T_C=W_T_C2,
                    detection_mode="aruco",
                ),
            ),
        ],
    )
    fleet_path = tmp_path / "fleet.yaml"
    save_fleet(fleet_path, fleet)

    out_dir = tmp_path / "consumer"
    paths = export_consumer_bundle(
        fleet_path=fleet_path,
        camera0_name="front-left",
        camera2_name="front-right",
        out_dir=out_dir,
    )

    np.testing.assert_allclose(_read_opencv_matrix(paths.c0_intrinsics, "K"), K0)
    np.testing.assert_allclose(_read_opencv_matrix(paths.c0_intrinsics, "D"), d0.reshape(1, 5))
    np.testing.assert_allclose(_read_opencv_matrix(paths.c2_intrinsics, "K"), K2)
    np.testing.assert_allclose(_read_opencv_matrix(paths.c2_intrinsics, "D"), d2.reshape(1, 5))

    expected_C2_T_C0 = se3.compose(se3.inverse(W_T_C2), W_T_C0)
    expected_R, expected_t = se3.to_Rt(expected_C2_T_C0)
    np.testing.assert_allclose(
        _read_opencv_matrix(paths.c0_to_c2_extrinsics, "R"),
        expected_R,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _read_opencv_matrix(paths.c0_to_c2_extrinsics, "T"),
        expected_t.reshape(3, 1),
        atol=1e-12,
    )

    pose_raw = yaml.safe_load(paths.camera0_pose.read_text())
    np.testing.assert_allclose(
        np.asarray(pose_raw["rotation_matrix"]["data"], dtype=np.float64).reshape(3, 3),
        np.eye(3),
    )
    np.testing.assert_allclose(
        np.asarray(pose_raw["translation_vector"]["data"], dtype=np.float64),
        np.zeros(3),
    )


def test_export_consumer_bundle_resolves_relative_paths_from_fleet(tmp_path):
    intrinsics_dir = tmp_path / "intrinsics"
    extrinsics_dir = tmp_path / "extrinsics"
    intrinsics_dir.mkdir()
    extrinsics_dir.mkdir()

    intr0_path = intrinsics_dir / "c0.yaml"
    intr2_path = intrinsics_dir / "c2.yaml"
    save_intrinsics(
        intr0_path,
        IntrinsicsRecord(
            K=np.eye(3),
            d=np.zeros(5),
            image_size=(1280, 720),
            reprojection_error_px=0.4,
            n_views=12,
        ),
    )
    save_intrinsics(
        intr2_path,
        IntrinsicsRecord(
            K=np.eye(3) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]),
            d=np.zeros(5),
            image_size=(1280, 720),
            reprojection_error_px=0.5,
            n_views=11,
        ),
    )

    board = CharucoConfig(
        squares_x=5,
        squares_y=7,
        square_size_m=0.03,
        marker_size_m=0.022,
        dictionary="DICT_5X5_100",
    )
    extr0_path = extrinsics_dir / "c0.yaml"
    extr2_path = extrinsics_dir / "c2.yaml"
    save_extrinsics(
        extr0_path,
        ExtrinsicsRecord(
            W_T_C=se3.identity(),
            board=board,
            n_frames=5,
            reprojection_error_px_mean=0.3,
            reprojection_error_px_max=0.6,
        ),
    )
    save_extrinsics(
        extr2_path,
        ExtrinsicsRecord(
            W_T_C=se3.from_Rt(np.eye(3), np.array([1.0, 0.0, 0.0])),
            board=board,
            n_frames=5,
            reprojection_error_px_mean=0.3,
            reprojection_error_px_max=0.6,
        ),
    )

    fleet = CameraFleetRecord(
        name="beta-rig",
        cameras=[
            FleetCamera(
                name="cam0",
                device={"serial": "S0"},
                intrinsics=FleetIntrinsics(path="intrinsics/c0.yaml", source="zhang"),
                extrinsics=FleetExtrinsics(path="extrinsics/c0.yaml"),
            ),
            FleetCamera(
                name="cam2",
                device={"serial": "S2"},
                intrinsics=FleetIntrinsics(path="intrinsics/c2.yaml", source="zhang"),
                extrinsics=FleetExtrinsics(path="extrinsics/c2.yaml"),
            ),
        ],
    )
    fleet_path = tmp_path / "fleet.yaml"
    save_fleet(fleet_path, fleet)

    out_dir = tmp_path / "bundle"
    paths = export_consumer_bundle(
        fleet_path=fleet_path,
        camera0_name="cam0",
        camera2_name="cam2",
        out_dir=out_dir,
    )
    np.testing.assert_allclose(
        _read_opencv_matrix(paths.c0_to_c2_extrinsics, "T"),
        np.array([[-1.0], [0.0], [0.0]]),
    )


def test_export_consumer_bundle_rejects_non_five_distortion(tmp_path):
    fleet = CameraFleetRecord(
        name="gamma-rig",
        cameras=[
            FleetCamera(
                name="cam0",
                intrinsics=FleetIntrinsics(
                    source="zhang",
                    image_size=(640, 480),
                    K=np.eye(3),
                    d=np.zeros(8),
                ),
                extrinsics=FleetExtrinsics(W_T_C=se3.identity()),
            ),
            FleetCamera(
                name="cam2",
                intrinsics=FleetIntrinsics(
                    source="zhang",
                    image_size=(640, 480),
                    K=np.eye(3),
                    d=np.zeros(5),
                ),
                extrinsics=FleetExtrinsics(W_T_C=se3.identity()),
            ),
        ],
    )
    fleet_path = tmp_path / "fleet.yaml"
    save_fleet(fleet_path, fleet)

    with np.testing.assert_raises(ValueError):
        export_consumer_bundle(
            fleet_path=fleet_path,
            camera0_name="cam0",
            camera2_name="cam2",
            out_dir=tmp_path / "bundle",
        )
