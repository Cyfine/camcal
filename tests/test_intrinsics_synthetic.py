"""Synthetic Zhang's-method recovery test.

Generate planar 3D points, project them through known camera intrinsics
+ extrinsics, then run ``calibrate_intrinsics`` and assert the recovered
K is within tolerance.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from camcal.intrinsics import calibrate_intrinsics


def _planar_grid(squares_x: int, squares_y: int, step_m: float) -> np.ndarray:
    """Generate (N, 3) corner positions for a planar grid in the target frame."""
    xs = np.arange(squares_x) * step_m
    ys = np.arange(squares_y) * step_m
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1)
    return pts.astype(np.float64)


def _project(
    object_points: np.ndarray,
    K: np.ndarray, d: np.ndarray,
    R: np.ndarray, t: np.ndarray,
    rng: np.random.Generator, noise_px: float,
) -> np.ndarray:
    """Project (N, 3) world points into the image, optionally with Gaussian noise."""
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec, t.reshape(3, 1),
        K, d,
    )
    img = projected.reshape(-1, 2)
    if noise_px > 0:
        img = img + rng.normal(scale=noise_px, size=img.shape)
    return img


def _view_poses(n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """A spread of (R, t) poses pointing the camera at the planar target."""
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n):
        # Camera ~ 0.3 m in front of the target, tilted up to ±25°.
        rotvec = rng.uniform(-1, 1, 3) * np.deg2rad(25)
        R = R_scipy.from_rotvec(rotvec).as_matrix()
        t = np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(0.25, 0.5),
        ])
        poses.append((R, t))
    return poses


def test_recover_intrinsics_noise_free():
    K_true = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    d_true = np.zeros(5)
    rng = np.random.default_rng(0)
    object_points_template = _planar_grid(8, 6, 0.025)
    poses = _view_poses(15, rng)

    image_points = []
    object_points = []
    for R, t in poses:
        img = _project(object_points_template, K_true, d_true, R, t, rng, 0.0)
        image_points.append(img)
        object_points.append(object_points_template.copy())

    result = calibrate_intrinsics(
        image_points=image_points,
        object_points=object_points,
        image_size=(640, 480),
    )

    # With zero noise + zero distortion the recovery should be near-exact.
    np.testing.assert_allclose(result.K, K_true, atol=1e-3, rtol=1e-4)
    assert result.reprojection_error_px < 1e-2
    assert len(result.per_view_C_T_T) == len(poses)


def test_recover_intrinsics_within_one_percent_with_noise():
    K_true = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    d_true = np.zeros(5)
    rng = np.random.default_rng(1)
    object_points_template = _planar_grid(8, 6, 0.025)
    poses = _view_poses(20, rng)

    image_points = []
    object_points = []
    for R, t in poses:
        img = _project(object_points_template, K_true, d_true, R, t, rng, noise_px=0.3)
        image_points.append(img)
        object_points.append(object_points_template.copy())

    result = calibrate_intrinsics(
        image_points=image_points,
        object_points=object_points,
        image_size=(640, 480),
    )
    fx, fy = result.K[0, 0], result.K[1, 1]
    cx, cy = result.K[0, 2], result.K[1, 2]
    assert abs(fx - 800.0) / 800.0 < 0.01
    assert abs(fy - 800.0) / 800.0 < 0.01
    assert abs(cx - 320.0) < 5.0
    assert abs(cy - 240.0) < 5.0
