import numpy as np

from camcal import se3
from camcal.charuco import CharucoConfig
from camcal.fleet import (
    CameraFleetRecord,
    FleetCamera,
    FleetExtrinsics,
    FleetIntrinsics,
    load_fleet,
    save_fleet,
)
from camcal.io_yaml import ExtrinsicsRecord, IntrinsicsRecord


def test_fleet_round_trip(tmp_path):
    intrinsics = IntrinsicsRecord(
        K=np.array([[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]),
        d=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
        image_size=(640, 480),
        reprojection_error_px=0.4,
        n_views=18,
        source="zhang",
        camera={"serial": "ABC123"},
    )
    extrinsics = ExtrinsicsRecord(
        W_T_C=se3.identity(),
        board=CharucoConfig(
            squares_x=5,
            squares_y=7,
            square_size_m=0.03,
            marker_size_m=0.022,
            dictionary="DICT_5X5_100",
        ),
        n_frames=6,
        reprojection_error_px_mean=0.3,
        reprojection_error_px_max=0.6,
        detection_mode="aruco",
        camera={"serial": "ABC123"},
    )
    fleet = CameraFleetRecord(
        name="alpha-rig",
        board_path="board.yaml",
        cameras=[
            FleetCamera(
                name="front-left",
                device={
                    "by_id_path": "/dev/v4l/by-id/usb-front-left",
                    "serial": "ABC123",
                },
                intrinsics=FleetIntrinsics.from_record(
                    intrinsics,
                    path="intrinsics/front-left.yaml",
                ),
                extrinsics=FleetExtrinsics.from_record(
                    extrinsics,
                    path="extrinsics/front-left.yaml",
                ),
            )
        ],
    )

    path = tmp_path / "fleet.yaml"
    save_fleet(path, fleet)
    loaded = load_fleet(path)
    assert loaded.name == "alpha-rig"
    assert loaded.board_path == "board.yaml"
    camera = loaded.get_camera("front-left")
    assert camera.device["serial"] == "ABC123"
    assert camera.intrinsics is not None
    assert camera.extrinsics is not None
    assert camera.intrinsics.path == "intrinsics/front-left.yaml"
    assert camera.extrinsics.path == "extrinsics/front-left.yaml"
    np.testing.assert_allclose(camera.intrinsics.K, intrinsics.K)
    np.testing.assert_allclose(camera.extrinsics.W_T_C, np.eye(4))


def test_fleet_upsert_camera_replaces_same_name():
    fleet = CameraFleetRecord(name="beta-rig")
    fleet.upsert_camera(FleetCamera(name="cam0", notes="first"))
    fleet.upsert_camera(FleetCamera(name="cam0", notes="second"))
    assert len(fleet.cameras) == 1
    assert fleet.get_camera("cam0").notes == "second"
