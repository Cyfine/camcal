from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from camcal import se3
from camcal.aruco import ArucoMarkerConfig, ArucoMarkerDetection, ArucoMarkerTarget
from camcal.charuco import aruco_dictionary
from camcal.pnp import solve_pnp


def test_detect_standalone_aruco_marker_on_rendered_image():
    config = ArucoMarkerConfig(
        dictionary="DICT_5X5_100",
        marker_id=7,
        size_m=0.1975,
    )
    target = ArucoMarkerTarget(config)

    marker = cv2.aruco.generateImageMarker(
        aruco_dictionary(config.dictionary),
        config.marker_id,
        240,
    )
    image = np.full((360, 360), 255, dtype=np.uint8)
    image[60:300, 60:300] = marker

    det = target.detect(image)

    assert det is not None
    assert det.marker_id == 7
    assert det.object_points.shape == (4, 3)
    assert det.image_points.shape == (4, 2)


def test_standalone_aruco_marker_pnp_uses_marker_corners():
    config = ArucoMarkerConfig(
        dictionary="DICT_5X5_100",
        marker_id=3,
        size_m=0.1975,
    )
    target = ArucoMarkerTarget(config)
    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    d = np.zeros(5)
    C_T_T_true = se3.from_Rt(
        R_scipy.from_rotvec(np.deg2rad([8.0, -5.0, 3.0])).as_matrix(),
        np.array([0.02, -0.01, 0.75]),
    )
    rvec, tvec = se3.to_rvec_tvec(C_T_T_true)
    projected, _ = cv2.projectPoints(
        target.object_points.reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        d,
    )
    det = ArucoMarkerDetection(
        object_points=target.object_points,
        image_points=projected.reshape(4, 2),
        marker_id=config.marker_id,
    )

    C_T_T = solve_pnp(
        det.object_points,
        det.image_points,
        K,
        d,
        method=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    assert C_T_T is not None
    assert se3.translation_distance(C_T_T, C_T_T_true) < 1e-4
    assert se3.rotation_distance(C_T_T, C_T_T_true) < 0.01
