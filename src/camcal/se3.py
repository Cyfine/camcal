"""SE(3) utilities.

Canonical in-memory form for SE(3) is a 4x4 NumPy float64 array of the form

    [[ R  t ]
     [ 0  1 ]]

where R is a 3x3 rotation matrix (orthonormal, det=+1) and t is a 3-vector
translation (meters).
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# ----- constants -----

_TOL_ORTHO = 1e-6
_TOL_DET = 1e-6
_TOL_LAST_ROW = 1e-9


# ----- validation -----


def is_valid_se3(T: np.ndarray, *, atol: float = _TOL_ORTHO) -> bool:
    """Return True iff T is a 4x4 SE(3) element within numerical tolerance."""
    if not isinstance(T, np.ndarray) or T.shape != (4, 4):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=_TOL_DET):
        return False
    return bool(
        np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=_TOL_LAST_ROW)
    )


def assert_se3(T: np.ndarray) -> None:
    """Raise ValueError if T is not a valid SE(3) matrix."""
    if not is_valid_se3(T):
        raise ValueError(f"Not a valid SE(3) matrix:\n{T}")


# ----- builders -----


def identity() -> np.ndarray:
    """4x4 identity SE(3)."""
    return np.eye(4, dtype=np.float64)


def from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build SE(3) from 3x3 rotation and 3-vector translation."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    assert_se3(T)
    return T


def project_to_se3(
    T: np.ndarray, *, max_orthogonality_error: float = 1e-3,
) -> np.ndarray:
    """Project a possibly-noisy 4x4 matrix onto SE(3) via SVD.

    Accepts inputs whose rotation block has numerical noise (typical of
    camera drivers), snaps it back onto SO(3), and returns a clean SE(3).
    Raises if the input rotation is too far from orthonormal — that
    indicates corruption rather than rounding noise.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"project_to_se3 expects 4x4, got {T.shape}")
    R_in = T[:3, :3]
    err = float(np.max(np.abs(R_in @ R_in.T - np.eye(3))))
    if err > max_orthogonality_error:
        raise ValueError(
            f"Rotation block is too far from orthonormal "
            f"(max |R R^T - I| = {err:.3e}, ceiling {max_orthogonality_error:.0e})."
        )
    U, _, Vt = np.linalg.svd(R_in)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        Vt[-1, :] *= -1
        R_proj = U @ Vt
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_proj
    out[:3, 3] = T[:3, 3]
    return out


def to_Rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose SE(3) into (R, t)."""
    assert_se3(T)
    return T[:3, :3].copy(), T[:3, 3].copy()


# ----- group operations -----


def inverse(T: np.ndarray) -> np.ndarray:
    """SE(3) inverse: rotation transpose + transformed translation."""
    assert_se3(T)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """T1 · T2."""
    assert_se3(T1)
    assert_se3(T2)
    return T1 @ T2


# ----- conversions: quaternion / rvec / tvec -----


def from_quat_t(quat_xyzw: Iterable[float], t: Iterable[float]) -> np.ndarray:
    """Build SE(3) from quaternion (x, y, z, w) + translation."""
    quat = np.asarray(list(quat_xyzw), dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(
            f"quat must be 4-vector (x,y,z,w), got shape {quat.shape}"
        )
    R = R_scipy.from_quat(quat).as_matrix()
    return from_Rt(R, t)


def to_quat_t(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose SE(3) into (quat_xyzw, translation)."""
    R, t = to_Rt(T)
    return R_scipy.from_matrix(R).as_quat(), t


def from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build SE(3) from OpenCV's (rvec, tvec) — Rodrigues rotation + translation."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return from_Rt(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose SE(3) into OpenCV (rvec, tvec)."""
    R, t = to_Rt(T)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3), t


# ----- distances and averaging -----


def translation_distance(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Euclidean distance between two SE(3) translations, in metres."""
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def rotation_distance(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Geodesic angle on SO(3) between two SE(3) rotations, in degrees."""
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def mean_se3(
    Ts: Iterable[np.ndarray], *, max_iter: int = 50, tol: float = 1e-9,
) -> np.ndarray:
    """Average a non-empty iterable of SE(3) elements.

    Translation: arithmetic mean. Rotation: iterative Karcher mean on
    SO(3) via the exponential/logarithm maps.
    """
    Ts_list = [np.asarray(T, dtype=np.float64) for T in Ts]
    if not Ts_list:
        raise ValueError("mean_se3 requires at least one SE(3) element")
    for T in Ts_list:
        assert_se3(T)

    t_mean = np.mean([T[:3, 3] for T in Ts_list], axis=0)

    Rs = [T[:3, :3] for T in Ts_list]
    R_mean = Rs[0].copy()
    for _ in range(max_iter):
        rotvecs = np.array(
            [R_scipy.from_matrix(R_mean.T @ R_i).as_rotvec() for R_i in Rs]
        )
        delta = rotvecs.mean(axis=0)
        if np.linalg.norm(delta) < tol:
            break
        R_mean = R_mean @ R_scipy.from_rotvec(delta).as_matrix()

    return from_Rt(R_mean, t_mean)


__all__ = [
    "assert_se3",
    "compose",
    "from_Rt",
    "from_quat_t",
    "from_rvec_tvec",
    "identity",
    "inverse",
    "is_valid_se3",
    "mean_se3",
    "project_to_se3",
    "rotation_distance",
    "to_Rt",
    "to_quat_t",
    "to_rvec_tvec",
    "translation_distance",
]
