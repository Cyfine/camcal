from __future__ import annotations

import cv2
import numpy as np
import yaml

from camcal.calibrate_fleet import main as solve_main
from camcal.fleet import CameraFleetRecord, FleetCamera, FleetIntrinsics, save_fleet
from camcal.marker_map import MarkerManifest, MarkerSpec, save_marker_manifest
from camcal.observe_fleet import main as observe_main


def _marker_image(path, *, x_shift: int) -> None:
    image = np.full((480, 640), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    for marker_id, x0 in ((0, 70 + x_shift), (9, 350 + x_shift)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 180)
        image[140:320, x0 : x0 + 180] = marker
    cv2.imwrite(str(path), image)


def test_offline_observe_solve_inspect_export_loop(tmp_path):
    K = np.array([[700.0, 0, 320.0], [0, 700.0, 240.0], [0, 0, 1.0]])
    cameras = []
    for name in ("a", "b"):
        cameras.append(
            FleetCamera(
                name=name,
                intrinsics=FleetIntrinsics(
                    source="test",
                    image_size=(640, 480),
                    K=K,
                    d=np.zeros(5),
                ),
            )
        )
    fleet_path = tmp_path / "fleet.yaml"
    save_fleet(fleet_path, CameraFleetRecord(name="smoke", cameras=cameras))
    marker_path = tmp_path / "markers.yaml"
    save_marker_manifest(
        marker_path,
        MarkerManifest(
            "DICT_6X6_250",
            (
                MarkerSpec(0, 0.20, measured_size_m=0.20),
                MarkerSpec(9, 0.20, measured_size_m=0.20),
            ),
        ),
    )
    table_marker_pose_path = tmp_path / "table-marker-pose.yaml"
    table_T_M9 = np.array(
        [
            [1.0, 0.0, 0.0, 0.1186],
            [0.0, 1.0, 0.0, -0.1186],
            [0.0, 0.0, 1.0, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    table_marker_pose_path.write_text(
        yaml.safe_dump({"T_T_M": table_T_M9.tolist()}, sort_keys=False)
    )
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _marker_image(image_a, x_shift=0)
    _marker_image(image_b, x_shift=20)
    observation_path = tmp_path / "observations.yaml"

    observe_code = observe_main(
        [
            "--fleet", str(fleet_path),
            "--markers", str(marker_path),
            "--out", str(observation_path),
            "--images", f"a={image_a}",
            "--images", f"b={image_b}",
        ]
    )
    result_path = tmp_path / "result.yaml"
    solve_code = solve_main(
        [
            "--fleet", str(fleet_path),
            "--markers", str(marker_path),
            "--observations", str(observation_path),
            "--out", str(result_path),
            "--room-anchor-marker-id", "0",
            "--table-marker-id", "9",
            "--table-marker-pose", str(table_marker_pose_path),
        ]
    )

    assert observe_code == 0
    assert solve_code == 0
    result = yaml.safe_load(result_path.read_text())
    assert result["solver_status"] == "SOLVED"
    assert set(result["cameras"]) == {"a", "b"}
    assert result["anchor_marker_id"] == 0
    assert "W_T_C maps camera-frame coordinates" in result["transform_convention"]
    np.testing.assert_allclose(
        result["cad_table_frame"]["T_T_M"],
        table_T_M9,
        atol=1e-12,
    )
    assert result["cad_table_frame"]["table_marker_id"] == 9
    room_T_M9 = np.asarray(result["markers"][9]["W_T_M"])
    expected_room_T_table = room_T_M9 @ np.linalg.inv(table_T_M9)
    np.testing.assert_allclose(
        result["cad_table_frame"]["R_T_T"],
        expected_room_T_table,
        atol=1e-12,
    )
    room_T_camera_a = np.asarray(result["cameras"]["a"]["W_T_C"])
    np.testing.assert_allclose(
        result["cad_table_frame"]["cameras"]["a"]["T_T_C"],
        np.linalg.inv(expected_room_T_table) @ room_T_camera_a,
        atol=1e-12,
    )
    assert set(result["cad_table_frame"]["cameras"]) == {"a", "b"}
    assert result_path.with_suffix(".html").is_file()
    solved_fleet = yaml.safe_load((tmp_path / "result_fleet.yaml").read_text())
    assert solved_fleet["world_frame"] == "cad_table"
    assert len(list((tmp_path / "result_residual_overlays").glob("*.png"))) == 2

    failed_result_path = tmp_path / "failed-result.yaml"
    failed_code = solve_main(
        [
            "--fleet", str(fleet_path),
            "--markers", str(marker_path),
            "--observations", str(observation_path),
            "--out", str(failed_result_path),
            "--room-anchor-marker-id", "0",
            "--table-marker-id", "9",
            "--table-marker-pose", str(table_marker_pose_path),
            "--max-evaluations", "1",
        ]
    )
    assert failed_code == 1
    assert yaml.safe_load(failed_result_path.read_text())["solver_status"] == "FAILED_TO_SOLVE"
    assert not (tmp_path / "failed-result_fleet.yaml").exists()
