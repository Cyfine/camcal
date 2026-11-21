"""Bundle adjustment of fixed cameras that observe one moving ChArUco board.

Unknowns: each camera's pose, the board's pose in every capture and,
optionally, some cameras' intrinsics (pinhole plus the five-coefficient
OpenCV distortion). Residuals: ChArUco corner reprojection errors in pixels,
under a robust loss.

The images fix the cameras only relative to each other and to the board. The
world is fixed by one of two gauges:

* a ``world`` capture: that board pose defines the world (full calibration);
* fixed camera poses: cameras that did not move keep their poses and so keep
  the world they already define (re-calibrating a camera that was bumped).

With a ``world`` capture the world frame is the board frame of that capture
turned about its x axis by 180 degrees: the origin is the board's outer
corner, x runs along the board's x axis, and z points out of the printed face.
A board lying face up on the floor therefore gives a z-up world.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from . import se3
from .board_observations import BoardObservation, BoardObservationDataset, board_corner_points

INTRINSIC_SIZE = 9  # fx, fy, cx, cy, k1, k2, p1, p2, k3
POSE_SIZE = 6  # rotation vector, translation
# Zhang initialisation needs views that differ in tilt; below this the prior is kept.
MIN_ZHANG_VIEWS = 6
# The world capture's board frame, turned so z leaves the printed face.
W_T_B_WORLD = se3.from_Rt(np.diag([1.0, -1.0, -1.0]), np.zeros(3))


@dataclass(frozen=True)
class ResidualStats:
    point_count: int
    mean_px: float
    rmse_px: float
    max_px: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point_count": self.point_count,
            "mean_px": self.mean_px,
            "rmse_px": self.rmse_px,
            "max_px": self.max_px,
        }


@dataclass(frozen=True)
class DroppedView:
    """A view left out because its corners do not fix the board's pose (too few, or nearly on one line)."""

    camera_name: str
    capture_id: str
    corner_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {"camera": self.camera_name, "capture": self.capture_id, "corners": self.corner_count}


@dataclass
class BoardBundleResult:
    """Solved cameras and board poses, with residual diagnostics."""

    K: dict[str, np.ndarray]
    d: dict[str, np.ndarray]
    W_T_C: dict[str, np.ndarray]
    W_T_B: dict[str, np.ndarray]
    free_intrinsics: tuple[str, ...]
    free_poses: tuple[str, ...]
    gauge: str
    camera_stats: dict[str, ResidualStats]
    observation_rmse_px: dict[tuple[str, str], float]
    initial_intrinsics: dict[str, str]
    zhang_rmse_px: dict[str, float] = field(default_factory=dict)
    success: bool = False
    message: str = ""
    evaluations: int = 0
    dropped_views: tuple[DroppedView, ...] = ()


def solve_board_bundle(
    dataset: BoardObservationDataset,
    *,
    free_intrinsics: Iterable[str] = (),
    fixed_poses: Mapping[str, np.ndarray] | None = None,
    initial_poses: Mapping[str, np.ndarray] | None = None,
    captures: Iterable[str] | None = None,
    loss: str = "huber",
    f_scale_px: float = 2.0,
    max_evaluations: int = 400,
) -> BoardBundleResult:
    """Solve camera poses, board poses and the selected cameras' intrinsics.

    ``fixed_poses`` maps camera name to ``W_T_C``. When it is empty the
    dataset's ``world`` capture fixes the world; otherwise those cameras do,
    and every other observed camera's pose is solved. ``initial_poses`` seeds
    free camera poses (for example the previous calibration).
    """
    fixed_poses = {name: np.asarray(T, dtype=np.float64) for name, T in (fixed_poses or {}).items()}
    capture_ids = list(captures) if captures is not None else dataset.capture_ids("solve", "world")
    used = [obs for obs in dataset.observations if obs.capture_id in set(capture_ids)]
    if not used:
        raise ValueError("no observations in the selected captures")
    observed = sorted({obs.camera_name for obs in used})
    free_intrinsics = tuple(sorted(set(free_intrinsics)))
    unknown = sorted(set(free_intrinsics) - set(observed))
    if unknown:
        raise ValueError(f"cameras with free intrinsics have no observations: {unknown}")
    unknown = sorted(set(fixed_poses) - {camera.name for camera in dataset.cameras})
    if unknown:
        raise ValueError(f"fixed poses name unknown cameras: {unknown}")

    if fixed_poses:
        gauge = "fixed cameras " + ", ".join(sorted(fixed_poses))
        world_capture = None
    else:
        world_capture = dataset.world_capture
        if world_capture is None or world_capture not in capture_ids:
            raise ValueError("a world capture is needed when no camera pose is fixed")
        gauge = f"world capture {world_capture}"

    snapshots = {camera.name: camera for camera in dataset.cameras}
    corners = board_corner_points(dataset.board)
    K: dict[str, np.ndarray] = {}
    d: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    zhang: dict[str, float] = {}
    for name in observed:
        K[name], d[name] = snapshots[name].K.copy(), _five_coefficients(snapshots[name].d, name)
        sources[name] = "prior"
    # Every view seeds the solve with its own board pose. A camera left with no view is reported
    # below as not linked to the world.
    dropped = tuple(
        DroppedView(obs.camera_name, obs.capture_id, len(obs.corner_ids))
        for obs in used
        if _try_pnp(K[obs.camera_name], d[obs.camera_name], corners[obs.corner_ids], obs.image_points) is None
    )
    left_out = {(view.camera_name, view.capture_id) for view in dropped}
    used = [obs for obs in used if (obs.camera_name, obs.capture_id) not in left_out]
    for name in observed:
        if name in free_intrinsics:
            views = [obs for obs in used if obs.camera_name == name]
            if len(views) >= MIN_ZHANG_VIEWS:
                K[name], d[name], zhang[name] = _zhang(views, corners, snapshots[name].image_size)
                sources[name] = "zhang"

    C_T_B = {
        (obs.camera_name, obs.capture_id): _pnp(K[obs.camera_name], d[obs.camera_name], corners[obs.corner_ids], obs.image_points)
        for obs in used
    }
    W_T_C, W_T_B = _initialise_poses(
        C_T_B,
        cameras=observed,
        captures=capture_ids,
        fixed_poses={name: T for name, T in fixed_poses.items() if name in observed},
        initial_poses=initial_poses or {},
        world_capture=world_capture,
    )

    free_poses = tuple(name for name in observed if name not in fixed_poses)
    free_captures = tuple(capture for capture in capture_ids if capture != world_capture and capture in W_T_B)
    layout = _Layout(observed, free_poses, free_intrinsics, free_captures)
    problem = _Problem(layout, used, corners, K, d, W_T_C, W_T_B)
    x0 = problem.initial_vector()
    solution = least_squares(
        problem.residuals,
        x0,
        jac_sparsity=problem.sparsity(),
        method="trf",
        loss=loss,
        f_scale=float(f_scale_px),
        x_scale="jac",
        max_nfev=int(max_evaluations),
    )
    K, d, W_T_C, W_T_B = problem.unpack(solution.x)
    errors = np.linalg.norm(problem.residuals(solution.x).reshape(-1, 2), axis=1)
    camera_stats, observation_rmse = _residual_tables(errors, problem.point_camera, problem.point_capture, observed, problem.captures)
    return BoardBundleResult(
        K=K,
        d=d,
        W_T_C=W_T_C,
        W_T_B=W_T_B,
        free_intrinsics=free_intrinsics,
        free_poses=free_poses,
        gauge=gauge,
        camera_stats=camera_stats,
        observation_rmse_px=observation_rmse,
        initial_intrinsics=sources,
        zhang_rmse_px=zhang,
        success=bool(solution.success),
        message=str(solution.message),
        evaluations=int(solution.nfev),
        dropped_views=dropped,
    )


def diagnose_rig(
    dataset: BoardObservationDataset,
    rig_poses: Mapping[str, np.ndarray],
    *,
    captures: Iterable[str] | None = None,
    loss: str = "huber",
    f_scale_px: float = 2.0,
) -> dict:
    """Which camera moved, and did its intrinsics change?

    First every camera keeps its current pose and intrinsics, and only the
    board poses are solved. Then each camera in turn has its pose freed. The
    camera that moved is the one whose freeing brings the error of all cameras
    down; an error that stays large whichever pose is freed points to changed
    intrinsics (for example a turned focus ring) or to more than one camera
    having moved.
    """
    capture_ids = list(captures) if captures is not None else dataset.capture_ids("solve", "world", "validation")
    observed = sorted({obs.camera_name for obs in dataset.observations if obs.capture_id in set(capture_ids)})
    poses = {name: np.asarray(rig_poses[name], dtype=np.float64) for name in observed if name in rig_poses}
    missing = sorted(set(observed) - set(poses))
    if missing:
        raise ValueError(f"cameras without a current pose: {missing}")
    common = dict(captures=capture_ids, loss=loss, f_scale_px=f_scale_px)
    baseline = solve_board_bundle(dataset, fixed_poses=poses, **common)
    report = {
        "captures": capture_ids,
        "baseline": {name: stats.as_dict() for name, stats in baseline.camera_stats.items()},
        "baseline_rmse_px": _overall_rmse(baseline.camera_stats),
        "dropped_views": [view.as_dict() for view in baseline.dropped_views],
        "freed": {},
    }
    if len(observed) < 2:
        return report
    for name in observed:
        others = {other: T for other, T in poses.items() if other != name}
        try:
            freed = solve_board_bundle(dataset, fixed_poses=others, initial_poses={name: poses[name]}, **common)
        except ValueError as error:
            report["freed"][name] = {"error": str(error)}
            continue
        report["freed"][name] = {
            "rmse_px": _overall_rmse(freed.camera_stats),
            "stats": {other: stats.as_dict() for other, stats in freed.camera_stats.items()},
            "translation_change_mm": 1000.0 * se3.translation_distance(poses[name], freed.W_T_C[name]),
            "rotation_change_deg": se3.rotation_distance(poses[name], freed.W_T_C[name]),
        }
    return report


def validate_transfer(
    dataset: BoardObservationDataset,
    K: Mapping[str, np.ndarray],
    d: Mapping[str, np.ndarray],
    W_T_C: Mapping[str, np.ndarray],
    *,
    captures: Iterable[str] | None = None,
) -> list[dict]:
    """Leave-one-camera-out transfer error on captures the solve did not use.

    For each capture and each camera that saw it together with another
    camera: the board pose is estimated from the other cameras alone, its
    corners are projected into the left-out camera, and the pixel error to
    that camera's detections is reported, also as millimetres at the board.
    """
    capture_ids = list(captures) if captures is not None else dataset.capture_ids("validation")
    corners = board_corner_points(dataset.board)
    by_capture: dict[str, list[BoardObservation]] = defaultdict(list)
    for obs in dataset.observations:
        if obs.capture_id in capture_ids and obs.camera_name in W_T_C:
            by_capture[obs.capture_id].append(obs)
    rows = []
    for capture_id in capture_ids:
        observations = by_capture.get(capture_id, [])
        for held_out in observations:
            others = [obs for obs in observations if obs is not held_out]
            if not others:
                continue
            try:
                W_T_B = _board_pose(others, corners, K, d, W_T_C)
            except ValueError:  # no other view fixes the board's pose
                continue
            name = held_out.camera_name
            C_T_W = se3.inverse(np.asarray(W_T_C[name]))
            points_c = _transform(se3.compose(C_T_W, W_T_B), corners[held_out.corner_ids])
            predicted = _project(points_c, _intrinsic_vector(K[name], d[name]))
            error_px = np.linalg.norm(predicted - held_out.image_points, axis=1)
            focal = 0.5 * (K[name][0, 0] + K[name][1, 1])
            error_mm = 1000.0 * error_px * points_c[:, 2] / focal
            rows.append(
                {
                    "capture": capture_id,
                    "camera": name,
                    "from": sorted(obs.camera_name for obs in others),
                    "points": int(error_px.size),
                    "rmse_px": float(np.sqrt(np.mean(error_px**2))),
                    "max_px": float(error_px.max()),
                    "rmse_mm": float(np.sqrt(np.mean(error_mm**2))),
                    "distance_m": float(np.median(points_c[:, 2])),
                }
            )
    return rows


def intrinsics_displacement_px(
    K_before: np.ndarray,
    d_before: np.ndarray,
    K_after: np.ndarray,
    d_after: np.ndarray,
    image_size: tuple[int, int],
    *,
    observed_points: np.ndarray | None = None,
    grid: tuple[int, int] = (32, 18),
) -> dict[str, float]:
    """How far, in pixels, the same viewing ray lands under the two models.

    Only pixels inside the convex hull of ``observed_points`` are compared when
    they are given: outside the corners that were seen, distortion is an
    extrapolation and its difference says nothing about the lens.
    """
    width, height = image_size
    u, v = np.meshgrid(np.linspace(0, width - 1, grid[0]), np.linspace(0, height - 1, grid[1]))
    pixels = np.column_stack((u.ravel(), v.ravel()))
    if observed_points is not None:
        hull = cv2.convexHull(np.asarray(observed_points, dtype=np.float32))
        inside = [cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in pixels]
        pixels = pixels[np.asarray(inside, dtype=bool)]
        if not len(pixels):
            return {"median_px": float("nan"), "max_px": float("nan")}
    rays = _unproject(pixels, _intrinsic_vector(K_after, d_after))
    before = _project(np.column_stack((rays, np.ones(len(rays)))), _intrinsic_vector(K_before, d_before))
    shift = np.linalg.norm(before - pixels, axis=1)
    return {"median_px": float(np.median(shift)), "max_px": float(shift.max())}


def image_coverage(points: np.ndarray, image_size: tuple[int, int], *, cells: tuple[int, int] = (8, 6)) -> float:
    """Fraction of a coarse image grid that holds at least one observed corner."""
    width, height = image_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return 0.0
    column = np.clip((points[:, 0] / width * cells[0]).astype(int), 0, cells[0] - 1)
    row = np.clip((points[:, 1] / height * cells[1]).astype(int), 0, cells[1] - 1)
    return len(set(zip(column.tolist(), row.tolist()))) / float(cells[0] * cells[1])


# ----- problem layout -----


@dataclass(frozen=True)
class _Layout:
    cameras: tuple[str, ...] | list[str]
    free_poses: tuple[str, ...]
    free_intrinsics: tuple[str, ...]
    free_captures: tuple[str, ...]

    def offsets(self) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
        cursor = 0
        poses, intrinsics, captures = {}, {}, {}
        for name in self.free_poses:
            poses[name] = cursor
            cursor += POSE_SIZE
        for name in self.free_intrinsics:
            intrinsics[name] = cursor
            cursor += INTRINSIC_SIZE
        for capture in self.free_captures:
            captures[capture] = cursor
            cursor += POSE_SIZE
        return poses, intrinsics, captures, cursor


class _Problem:
    def __init__(self, layout, observations, corners, K, d, W_T_C, W_T_B):
        self.layout = layout
        self.cameras = list(layout.cameras)
        self.captures = sorted(W_T_B)
        self.pose_offset, self.intrinsic_offset, self.capture_offset, self.size = layout.offsets()
        camera_index = {name: i for i, name in enumerate(self.cameras)}
        capture_index = {capture: i for i, capture in enumerate(self.captures)}
        kept = [obs for obs in observations if obs.capture_id in capture_index]
        self.point_camera = np.concatenate([np.full(len(obs.corner_ids), camera_index[obs.camera_name]) for obs in kept])
        self.point_capture = np.concatenate([np.full(len(obs.corner_ids), capture_index[obs.capture_id]) for obs in kept])
        self.board_points = np.concatenate([corners[obs.corner_ids] for obs in kept])
        self.image_points = np.concatenate([obs.image_points for obs in kept])
        self.fixed_intrinsics = np.array([_intrinsic_vector(K[name], d[name]) for name in self.cameras])
        self.fixed_camera = np.array([_pose_vector(se3.inverse(W_T_C[name])) for name in self.cameras])
        self.fixed_board = np.array([_pose_vector(W_T_B[capture]) for capture in self.captures])

    def initial_vector(self) -> np.ndarray:
        x = np.zeros(self.size)
        for name, offset in self.pose_offset.items():
            x[offset : offset + POSE_SIZE] = self.fixed_camera[self.cameras.index(name)]
        for name, offset in self.intrinsic_offset.items():
            x[offset : offset + INTRINSIC_SIZE] = self.fixed_intrinsics[self.cameras.index(name)]
        for capture, offset in self.capture_offset.items():
            x[offset : offset + POSE_SIZE] = self.fixed_board[self.captures.index(capture)]
        return x

    def _parameters(self, x):
        camera = self.fixed_camera.copy()
        intrinsics = self.fixed_intrinsics.copy()
        board = self.fixed_board.copy()
        for name, offset in self.pose_offset.items():
            camera[self.cameras.index(name)] = x[offset : offset + POSE_SIZE]
        for name, offset in self.intrinsic_offset.items():
            intrinsics[self.cameras.index(name)] = x[offset : offset + INTRINSIC_SIZE]
        for capture, offset in self.capture_offset.items():
            board[self.captures.index(capture)] = x[offset : offset + POSE_SIZE]
        return camera, intrinsics, board

    def residuals(self, x) -> np.ndarray:
        camera, intrinsics, board = self._parameters(x)
        R_cw = Rotation.from_rotvec(camera[:, :3]).as_matrix()
        R_wb = Rotation.from_rotvec(board[:, :3]).as_matrix()
        b, c = self.point_capture, self.point_camera
        points_w = np.einsum("nij,nj->ni", R_wb[b], self.board_points) + board[b, 3:]
        points_c = np.einsum("nij,nj->ni", R_cw[c], points_w) + camera[c, 3:]
        return (_project(points_c, intrinsics[c]) - self.image_points).ravel()

    def sparsity(self) -> coo_matrix:
        rows, columns = [], []
        point_rows = np.arange(len(self.point_camera))
        blocks = [
            (self.pose_offset, self.cameras, self.point_camera, POSE_SIZE),
            (self.intrinsic_offset, self.cameras, self.point_camera, INTRINSIC_SIZE),
            (self.capture_offset, self.captures, self.point_capture, POSE_SIZE),
        ]
        for offsets, names, index, width in blocks:
            for name, offset in offsets.items():
                selected = point_rows[index == names.index(name)]
                for axis in (0, 1):
                    for k in range(width):
                        rows.append(2 * selected + axis)
                        columns.append(np.full(selected.size, offset + k))
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        columns = np.concatenate(columns) if columns else np.zeros(0, int)
        return coo_matrix((np.ones(rows.size), (rows, columns)), shape=(2 * len(self.point_camera), self.size))

    def unpack(self, x):
        camera, intrinsics, board = self._parameters(x)
        K, d, W_T_C, W_T_B = {}, {}, {}, {}
        for i, name in enumerate(self.cameras):
            K[name], d[name] = _intrinsic_matrices(intrinsics[i])
            W_T_C[name] = se3.inverse(_pose_matrix(camera[i]))
        for i, capture in enumerate(self.captures):
            W_T_B[capture] = _pose_matrix(board[i])
        return K, d, W_T_C, W_T_B


# ----- initialisation -----


def _initialise_poses(C_T_B, *, cameras, captures, fixed_poses, initial_poses, world_capture):
    """Chain single-view poses outward from the gauge, averaging where views agree."""
    W_T_C = dict(fixed_poses)
    W_T_B = {world_capture: W_T_B_WORLD.copy()} if world_capture is not None else {}
    seeded = {name: np.asarray(T, dtype=np.float64) for name, T in initial_poses.items() if name in cameras}
    changed = True
    while changed:
        changed = False
        for capture in captures:
            if capture in W_T_B:
                continue
            estimates = [se3.compose(W_T_C[name], C_T_B[name, capture]) for name in W_T_C if (name, capture) in C_T_B]
            if estimates:
                W_T_B[capture] = se3.mean_se3(estimates)
                changed = True
        for name in cameras:
            if name in W_T_C:
                continue
            if name in seeded:
                W_T_C[name] = seeded[name]
                changed = True
                continue
            estimates = [se3.compose(W_T_B[capture], se3.inverse(C_T_B[name, capture])) for capture in W_T_B if (name, capture) in C_T_B]
            if estimates:
                W_T_C[name] = se3.mean_se3(estimates)
                changed = True
    unreached = sorted(set(cameras) - set(W_T_C))
    if unreached:
        raise ValueError(f"cameras not linked to the world through shared captures: {unreached}")
    return W_T_C, W_T_B


def _zhang(views: list[BoardObservation], corners: np.ndarray, image_size) -> tuple[np.ndarray, np.ndarray, float]:
    objects = [corners[obs.corner_ids].astype(np.float32) for obs in views]
    images = [obs.image_points.astype(np.float32) for obs in views]
    rms, K, d, _, _ = cv2.calibrateCamera(objects, images, tuple(int(v) for v in image_size), None, None)
    return np.asarray(K, dtype=np.float64), np.asarray(d, dtype=np.float64).reshape(-1)[:5], float(rms)


def _pnp(K, d, object_points, image_points) -> np.ndarray:
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, d, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("board pose could not be initialised from one view")
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, d, rvec, tvec)
    return se3.from_rvec_tvec(rvec, tvec)


def _try_pnp(K, d, object_points, image_points) -> np.ndarray | None:
    """The single-view board pose, or None when the corners do not fix it."""
    try:
        return _pnp(K, d, object_points, image_points)
    except (ValueError, cv2.error):
        return None


def _board_pose(observations, corners, K, d, W_T_C) -> np.ndarray:
    """Board pose seen by fixed, calibrated cameras."""
    for first in observations:
        name = first.camera_name
        C_T_B = _try_pnp(K[name], d[name], corners[first.corner_ids], first.image_points)
        if C_T_B is not None:
            break
    else:
        raise ValueError("no view fixes the board pose")
    initial = se3.compose(W_T_C[name], C_T_B)
    C_T_W = {obs.camera_name: se3.inverse(np.asarray(W_T_C[obs.camera_name])) for obs in observations}

    def residuals(x):
        W_T_B = _pose_matrix(x)
        out = []
        for obs in observations:
            points_c = _transform(se3.compose(C_T_W[obs.camera_name], W_T_B), corners[obs.corner_ids])
            out.append(_project(points_c, _intrinsic_vector(K[obs.camera_name], d[obs.camera_name])) - obs.image_points)
        return np.concatenate(out).ravel()

    return _pose_matrix(least_squares(residuals, _pose_vector(initial), method="lm").x)


# ----- geometry -----


def _project(points_c: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """OpenCV pinhole projection with k1, k2, p1, p2, k3; ``intrinsics`` is (9,) or (N, 9)."""
    q = np.broadcast_to(intrinsics, (len(points_c), INTRINSIC_SIZE))
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = q.T
    z = np.maximum(points_c[:, 2], 1e-9)
    x, y = points_c[:, 0] / z, points_c[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack((fx * xd + cx, fy * yd + cy))


def _unproject(pixels: np.ndarray, intrinsics: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Normalised image coordinates of pixels: the fixed-point inverse of :func:`_project`."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics
    xd = (pixels[:, 0] - cx) / fx
    yd = (pixels[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x = (xd - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / radial
        y = (yd - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / radial
    return np.column_stack((x, y))


def _transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def _five_coefficients(d: np.ndarray, name: str) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.size > 5 and np.any(np.abs(d[5:]) > 0):
        raise ValueError(f"camera {name!r}: only the five-coefficient distortion model is supported")
    return np.pad(d[:5], (0, max(0, 5 - d.size)))


def _intrinsic_vector(K: np.ndarray, d: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    return np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], _five_coefficients(d, "")]


def _intrinsic_matrices(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = vector[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), np.asarray(vector[4:], dtype=np.float64).copy()


def _pose_vector(T: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _pose_matrix(vector: np.ndarray) -> np.ndarray:
    return se3.from_Rt(Rotation.from_rotvec(vector[:3]).as_matrix(), vector[3:6])


def _residual_tables(errors, point_camera, point_capture, cameras, captures):
    camera_stats = {}
    for i, name in enumerate(cameras):
        values = errors[point_camera == i]
        if values.size:
            camera_stats[name] = _stats(values)
    observation_rmse = {}
    for i, name in enumerate(cameras):
        for j, capture in enumerate(captures):
            values = errors[(point_camera == i) & (point_capture == j)]
            if values.size:
                observation_rmse[name, capture] = float(np.sqrt(np.mean(values**2)))
    return camera_stats, observation_rmse


def _overall_rmse(camera_stats: Mapping[str, ResidualStats]) -> float:
    squares = sum(stats.rmse_px**2 * stats.point_count for stats in camera_stats.values())
    count = sum(stats.point_count for stats in camera_stats.values())
    return float(np.sqrt(squares / count)) if count else float("nan")


def _stats(values: np.ndarray) -> ResidualStats:
    return ResidualStats(
        point_count=int(values.size),
        mean_px=float(values.mean()),
        rmse_px=float(np.sqrt(np.mean(values**2))),
        max_px=float(values.max()),
    )


__all__ = [
    "BoardBundleResult",
    "ResidualStats",
    "W_T_B_WORLD",
    "diagnose_rig",
    "image_coverage",
    "intrinsics_displacement_px",
    "solve_board_bundle",
    "validate_transfer",
]
