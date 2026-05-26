"""Perspective-n-Point — recover target pose in camera frame from 2D-3D correspondences.

The output ``C_T_T`` is the pose of the target expressed in the camera
frame (OpenCV's ``(rvec, tvec)`` convention).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import se3

DEFAULT_METHOD = cv2.SOLVEPNP_ITERATIVE


def _validate_pnp_inputs(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
) -> None:
    """Strict input validation. Raises :class:`ValueError` on any anomaly."""
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError(
            f"object_points must be (N, 3), got shape {object_points.shape}"
        )
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError(
            f"image_points must be (N, 2), got shape {image_points.shape}"
        )
    if object_points.shape[0] != image_points.shape[0]:
        raise ValueError(
            f"point count mismatch: {object_points.shape[0]} object vs "
            f"{image_points.shape[0]} image"
        )
    if object_points.shape[0] < 4:
        raise ValueError(
            f"PnP requires at least 4 points, got {object_points.shape[0]}"
        )
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    for name, arr in (
        ("object_points", object_points),
        ("image_points", image_points),
        ("K", K),
        ("distortion coefficients", d),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
    *,
    method: int = DEFAULT_METHOD,
) -> np.ndarray | None:
    """Solve PnP and return ``C_T_T`` (target pose in camera frame).

    Parameters
    ----------
    object_points
        ``(N, 3)`` array of 3D points in the target frame.
    image_points
        ``(N, 2)`` array of corresponding 2D pixel coordinates.
    K
        ``3x3`` camera intrinsic matrix.
    d
        Distortion coefficients (5 or 8 elements).
    method
        ``cv2.SOLVEPNP_*`` method. Default ``SOLVEPNP_ITERATIVE``.

    Returns
    -------
    A ``4x4`` SE(3) ``C_T_T`` if PnP succeeded, else ``None``.

    Raises
    ------
    ValueError
        For input shape, count, or NaN/inf anomalies. Geometry failures
        (PnP cannot find a solution) return ``None``.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).flatten()

    _validate_pnp_inputs(object_points, image_points, K, d)

    obj = object_points.astype(np.float32).reshape(-1, 1, 3)
    img = image_points.astype(np.float32).reshape(-1, 1, 2)

    success, rvec, tvec = cv2.solvePnP(obj, img, K, d, flags=method)
    if not success:
        return None
    return se3.from_rvec_tvec(rvec, tvec)


def reprojection_error_px(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
    C_T_T: np.ndarray,
) -> tuple[float, float]:
    """Project ``object_points`` through ``C_T_T`` and return ``(mean, max)`` pixel error."""
    rvec, tvec = se3.to_rvec_tvec(C_T_T)
    projected, _ = cv2.projectPoints(
        object_points.astype(np.float64).reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K.astype(np.float64),
        d.astype(np.float64).flatten(),
    )
    diff = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    per_pt = np.linalg.norm(diff, axis=1)
    return float(per_pt.mean()), float(per_pt.max())


__all__ = ["DEFAULT_METHOD", "reprojection_error_px", "solve_pnp"]
