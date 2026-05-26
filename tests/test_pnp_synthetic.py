"""Synthetic PnP recovery test.

Project known 3D points through a known camera pose, then solve PnP and
assert recovery within tight tolerances.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from camcal import se3
from camcal.pnp import reprojection_error_px, solve_pnp


def _planar_grid() -> np.ndarray:
    xs = np.arange(8) * 0.025
    ys = np.arange(6) * 0.025
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1)


def _project(
    object_points: np.ndarray, C_T_T: np.ndarray,
    K: np.ndarray, d: np.ndarray,
    rng: np.random.Generator | None = None, noise_px: float = 0.0,
) -> np.ndarray:
    rvec, tvec = se3.to_rvec_tvec(C_T_T)
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec.reshape(3, 1), tvec.reshape(3, 1),
        K, d,
    )
    img = projected.reshape(-1, 2)
    if rng is not None and noise_px > 0:
        img = img + rng.normal(scale=noise_px, size=img.shape)
    return img


def _make_pose(rotvec_deg: tuple[float, float, float], t: tuple[float, float, float]) -> np.ndarray:
    R = R_scipy.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()
    return se3.from_Rt(R, np.asarray(t))


def test_pnp_noise_free_recovers_pose():
    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    d = np.zeros(5)
    C_T_T_true = _make_pose((10, -8, 5), (0.02, -0.01, 0.40))

    obj = _planar_grid()
    img = _project(obj, C_T_T_true, K, d)
    C_T_T = solve_pnp(obj, img, K, d)
    assert C_T_T is not None

    # Translation within 0.1 mm, rotation within 0.01 deg.
    assert se3.translation_distance(C_T_T, C_T_T_true) < 1e-4
    assert se3.rotation_distance(C_T_T, C_T_T_true) < 0.01


def test_pnp_with_pixel_noise_within_tolerance():
    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    d = np.zeros(5)
    C_T_T_true = _make_pose((-12, 6, -3), (-0.04, 0.03, 0.45))
    rng = np.random.default_rng(0)
    obj = _planar_grid()
    img = _project(obj, C_T_T_true, K, d, rng, noise_px=0.3)
    C_T_T = solve_pnp(obj, img, K, d)
    assert C_T_T is not None
    # 0.3 px noise across 48 points on a planar target — sub-mm
    # translation, sub-half-degree rotation (planar PnP has known
    # rotation conditioning at small tilts).
    assert se3.translation_distance(C_T_T, C_T_T_true) < 1e-3
    assert se3.rotation_distance(C_T_T, C_T_T_true) < 0.5


def test_reprojection_error_helper():
    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    d = np.zeros(5)
    C_T_T = _make_pose((5, 5, 5), (0.01, 0.02, 0.40))
    obj = _planar_grid()
    img = _project(obj, C_T_T, K, d)
    mean, mx = reprojection_error_px(
        object_points=obj, image_points=img, K=K, d=d, C_T_T=C_T_T,
    )
    assert mean < 1e-3
    assert mx < 1e-3
