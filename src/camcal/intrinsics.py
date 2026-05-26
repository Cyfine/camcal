"""Camera intrinsic calibration via Zhang's method.

Wraps ``cv2.calibrateCamera``. The recovered ``K`` and per-view
extrinsics are the basis for every downstream pose computation, so the
module validates inputs strictly and surfaces every detected anomaly as
a :class:`ValueError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import se3

# Minimum views Zhang's method needs to constrain the IAC for K's
# 5 intrinsic parameters.
MIN_VIEWS = 3


@dataclass
class IntrinsicsResult:
    """Output of :func:`calibrate_intrinsics`."""

    K: np.ndarray                       # (3, 3)
    d: np.ndarray                       # distortion coefficients (5 or 8)
    image_size: tuple[int, int]         # (width, height) in pixels
    reprojection_error_px: float
    per_view_C_T_T: list[np.ndarray] = field(repr=False)  # one SE(3) per view


def _validate_inputs(
    image_points: Sequence[np.ndarray],
    object_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> None:
    """Strict input validation. Raises :class:`ValueError` on any anomaly."""
    if len(image_points) != len(object_points):
        raise ValueError(
            f"image_points and object_points must be same length, "
            f"got {len(image_points)} vs {len(object_points)}"
        )
    n_views = len(image_points)
    if n_views < MIN_VIEWS:
        raise ValueError(
            f"calibrate_intrinsics requires at least {MIN_VIEWS} views, got {n_views}"
        )
    w, h = image_size
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError(
            f"image_size must be positive (width, height) ints, got {image_size}"
        )
    for i, (ip, op) in enumerate(zip(image_points, object_points, strict=True)):
        ip_arr = np.asarray(ip)
        op_arr = np.asarray(op)
        if ip_arr.ndim != 2 or ip_arr.shape[1] != 2:
            raise ValueError(
                f"view {i}: image_points must be (N, 2), got shape {ip_arr.shape}"
            )
        if op_arr.ndim != 2 or op_arr.shape[1] != 3:
            raise ValueError(
                f"view {i}: object_points must be (N, 3), got shape {op_arr.shape}"
            )
        if ip_arr.shape[0] != op_arr.shape[0]:
            raise ValueError(
                f"view {i}: point count mismatch — "
                f"{ip_arr.shape[0]} image vs {op_arr.shape[0]} object"
            )
        if ip_arr.shape[0] < 4:
            raise ValueError(
                f"view {i}: need at least 4 points per view, got {ip_arr.shape[0]}"
            )
        if not np.all(np.isfinite(ip_arr)):
            raise ValueError(f"view {i}: image_points contains NaN or inf")
        if not np.all(np.isfinite(op_arr)):
            raise ValueError(f"view {i}: object_points contains NaN or inf")


def calibrate_intrinsics(
    image_points: Sequence[np.ndarray],
    object_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    flags: int = 0,
) -> IntrinsicsResult:
    """Recover camera intrinsics from N views of a planar calibration target.

    Parameters
    ----------
    image_points
        List of ``(N_i, 2)`` arrays — detected target corners in pixels.
    object_points
        List of ``(N_i, 3)`` arrays — corresponding 3D points in the
        target frame. For planar targets the z-column is zero.
    image_size
        ``(width, height)`` in pixels.
    flags
        Optional ``cv2.CALIB_*`` flags. Default ``0`` (full optimisation).

    Returns
    -------
    :class:`IntrinsicsResult` with ``K``, ``d``, per-view extrinsics
    ``C_T_T``, and the global reprojection error.

    Notes
    -----
    Per-view extrinsics are returned in the OpenCV convention: target →
    camera, i.e. ``C_T_T`` in the ``A_T_B = transform of B in A`` form.
    """
    _validate_inputs(image_points, object_points, image_size)

    img_pts_f = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points]
    obj_pts_f = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in object_points]

    rms, K, d, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_f,
        img_pts_f,
        image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
    )

    per_view_C_T_T = [
        se3.from_rvec_tvec(rvec, tvec)
        for rvec, tvec in zip(rvecs, tvecs, strict=True)
    ]

    return IntrinsicsResult(
        K=np.asarray(K, dtype=np.float64),
        d=np.asarray(d, dtype=np.float64).flatten(),
        per_view_C_T_T=per_view_C_T_T,
        reprojection_error_px=float(rms),
        image_size=(int(image_size[0]), int(image_size[1])),
    )


__all__ = ["IntrinsicsResult", "calibrate_intrinsics", "MIN_VIEWS"]
