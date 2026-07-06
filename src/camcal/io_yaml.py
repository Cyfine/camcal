"""Flat, human-editable YAML schemas for intrinsics and extrinsics.

Two record types, each with a ``save_*`` / ``load_*`` pair:

* ``intrinsics.yaml`` — :class:`IntrinsicsRecord` (``K``, ``d``,
  ``image_size``, plus provenance).
* ``extrinsics.yaml`` — :class:`ExtrinsicsRecord` (``W_T_C`` plus the
  target snapshot the world frame is pinned to).

The module also knows how to read the repo's RealSense factory-intrinsics
bundle format and resolve one camera record from it for downstream
extrinsics work.

The schemas are intentionally flat: numbers, lists, and strings, no
nested object classes. Anything you load is plain ``numpy`` / Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import se3
from .charuco import CharucoConfig


# ----- intrinsics -----


@dataclass
class IntrinsicsRecord:
    """In-memory mirror of ``intrinsics.yaml``."""

    K: np.ndarray                       # (3, 3)
    d: np.ndarray                       # (5,) or (8,)
    image_size: tuple[int, int]
    reprojection_error_px: float
    n_views: int
    source: str = "zhang"
    created_at: str = ""                # ISO-8601 UTC; auto-filled on save
    notes: str = ""
    camera: dict[str, Any] | None = None  # provenance of the camera that produced this


@dataclass
class RealSenseIntrinsicsCameraRecord:
    """One camera entry from ``realsense_intrinsics.yaml``."""

    serial: str
    model: str
    product_id: str
    firmware_version: str
    distortion_model: str
    K: np.ndarray
    d: np.ndarray


@dataclass
class RealSenseIntrinsicsBundle:
    """Repo-local bundle of RealSense factory intrinsics."""

    source: str
    created_at: str
    profile: dict[str, Any]
    cameras: list[RealSenseIntrinsicsCameraRecord]


def save_intrinsics(path: str | Path, record: IntrinsicsRecord) -> None:
    """Write an :class:`IntrinsicsRecord` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "K": _matrix_to_list(record.K, expected_shape=(3, 3)),
        "d": [float(x) for x in np.asarray(record.d).flatten()],
        "image_size": [int(record.image_size[0]), int(record.image_size[1])],
        "reprojection_error_px": float(record.reprojection_error_px),
        "n_views": int(record.n_views),
        "source": str(record.source),
        "created_at": record.created_at or _utc_now_iso(),
    }
    if record.camera is not None:
        body["camera"] = record.camera
    if record.notes:
        body["notes"] = record.notes
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_intrinsics(path: str | Path) -> IntrinsicsRecord:
    """Read an :class:`IntrinsicsRecord` from ``path``."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    K = np.asarray(raw["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path}: K must be 3x3, got shape {K.shape}")
    d = np.asarray(raw["d"], dtype=np.float64).flatten()
    image_size = tuple(int(x) for x in raw["image_size"])
    if len(image_size) != 2:
        raise ValueError(
            f"{path}: image_size must be [width, height], got {raw['image_size']!r}"
        )
    return IntrinsicsRecord(
        K=K,
        d=d,
        image_size=(image_size[0], image_size[1]),
        reprojection_error_px=float(raw["reprojection_error_px"]),
        n_views=int(raw["n_views"]),
        source=str(raw.get("source", "zhang")),
        created_at=str(raw.get("created_at", "")),
        notes=str(raw.get("notes", "")),
        camera=raw.get("camera"),
    )


def load_realsense_intrinsics_bundle(
    path: str | Path,
) -> RealSenseIntrinsicsBundle:
    """Read a repo-local ``realsense_intrinsics.yaml`` bundle."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw.get("source") != "realsense_api" or "cameras" not in raw:
        raise ValueError(f"{path}: not a RealSense factory-intrinsics bundle")
    profile = dict(raw.get("profile") or {})
    cameras: list[RealSenseIntrinsicsCameraRecord] = []
    for index, cam_raw in enumerate(raw["cameras"]):
        K = np.asarray(cam_raw["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(
                f"{path}: camera[{index}] K must be 3x3, got shape {K.shape}"
            )
        d = np.asarray(cam_raw["d"], dtype=np.float64).flatten()
        cameras.append(
            RealSenseIntrinsicsCameraRecord(
                serial=str(cam_raw["serial"]),
                model=str(cam_raw.get("model", "")),
                product_id=str(cam_raw.get("product_id", "")),
                firmware_version=str(cam_raw.get("firmware_version", "")),
                distortion_model=str(cam_raw.get("distortion_model", "")),
                K=K,
                d=d,
            )
        )
    return RealSenseIntrinsicsBundle(
        source=str(raw.get("source", "realsense_api")),
        created_at=str(raw.get("created_at", "")),
        profile=profile,
        cameras=cameras,
    )


def resolve_intrinsics_file(
    path: str | Path,
    *,
    camera_serial: str | None = None,
) -> tuple[IntrinsicsRecord, dict[str, Any] | None]:
    """Load either a single-camera intrinsics file or a RealSense bundle.

    Returns ``(record, profile)`` where ``profile`` is only populated for the
    RealSense bundle case.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if "K" in raw:
        return load_intrinsics(path), None
    bundle = load_realsense_intrinsics_bundle(path)
    camera = _select_realsense_camera(bundle, camera_serial=camera_serial)
    image_size_raw = profile_image_size(bundle.profile)
    return (
        IntrinsicsRecord(
            K=camera.K,
            d=camera.d,
            image_size=image_size_raw,
            reprojection_error_px=0.0,
            n_views=0,
            source="realsense_api",
            created_at=bundle.created_at,
            camera={
                "serial": camera.serial,
                "model": camera.model,
                "product_id": camera.product_id,
                "firmware_version": camera.firmware_version,
                "distortion_model": camera.distortion_model,
                "source": "realsense_api",
            },
        ),
        bundle.profile,
    )


def profile_image_size(profile: dict[str, Any]) -> tuple[int, int]:
    """Extract ``(width, height)`` from a RealSense profile block."""
    image_size = tuple(int(x) for x in profile.get("image_size", []))
    if len(image_size) != 2:
        raise ValueError(
            f"profile.image_size must be [width, height], got "
            f"{profile.get('image_size')!r}"
        )
    return image_size[0], image_size[1]


def _select_realsense_camera(
    bundle: RealSenseIntrinsicsBundle,
    *,
    camera_serial: str | None,
) -> RealSenseIntrinsicsCameraRecord:
    if camera_serial is not None:
        for camera in bundle.cameras:
            if camera.serial == camera_serial:
                return camera
        raise ValueError(
            f"RealSense intrinsics bundle has no camera with serial "
            f"{camera_serial!r}"
        )
    if len(bundle.cameras) == 1:
        return bundle.cameras[0]
    raise ValueError(
        "RealSense intrinsics bundle contains multiple cameras; provide "
        "--camera-serial or use a single-camera intrinsics file."
    )


# ----- extrinsics -----


@dataclass
class ExtrinsicsRecord:
    """In-memory mirror of ``extrinsics.yaml``.

    ``W_T_C`` is the pose of the camera in the world frame. The world
    frame is defined by the target snapshot in ``target``. Older ChArUco
    records also carry the same data in ``board`` for compatibility.
    """

    W_T_C: np.ndarray                   # (4, 4) SE(3)
    n_frames: int
    reprojection_error_px_mean: float
    reprojection_error_px_max: float
    board: CharucoConfig | None = None
    target: dict[str, Any] | None = None
    intrinsics_path: str = ""
    detection_mode: str = "charuco"
    created_at: str = ""
    notes: str = ""
    camera: dict[str, Any] | None = None  # provenance of the camera that produced this
    per_frame_W_T_C: list[np.ndarray] = field(default_factory=list, repr=False)


def save_extrinsics(path: str | Path, record: ExtrinsicsRecord) -> None:
    """Write an :class:`ExtrinsicsRecord` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    se3.assert_se3(record.W_T_C)
    quat_xyzw, t = se3.to_quat_t(record.W_T_C)
    target = record.target or _target_from_board(record.board)
    body: dict[str, Any] = {
        "W_T_C": _matrix_to_list(record.W_T_C, expected_shape=(4, 4)),
        "translation_m": [float(x) for x in t],
        "quaternion_xyzw": [float(x) for x in quat_xyzw],
        "n_frames": int(record.n_frames),
        "reprojection_error_px_mean": float(record.reprojection_error_px_mean),
        "reprojection_error_px_max": float(record.reprojection_error_px_max),
        "target": target,
        "intrinsics_path": str(record.intrinsics_path),
        "detection_mode": str(record.detection_mode),
        "created_at": record.created_at or _utc_now_iso(),
    }
    if record.board is not None:
        body["board"] = _board_to_dict(record.board)
    if record.camera is not None:
        body["camera"] = record.camera
    if record.notes:
        body["notes"] = record.notes
    if record.per_frame_W_T_C:
        body["per_frame_W_T_C"] = [
            _matrix_to_list(T, expected_shape=(4, 4))
            for T in record.per_frame_W_T_C
        ]
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_extrinsics(path: str | Path) -> ExtrinsicsRecord:
    """Read an :class:`ExtrinsicsRecord` from ``path``."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    W_T_C = np.asarray(raw["W_T_C"], dtype=np.float64)
    if W_T_C.shape != (4, 4):
        raise ValueError(f"{path}: W_T_C must be 4x4, got shape {W_T_C.shape}")
    se3.assert_se3(W_T_C)
    target = raw.get("target")
    board = _board_from_extrinsics_raw(raw)
    if target is None and board is not None:
        target = _target_from_board(board)
    per_frame_raw = raw.get("per_frame_W_T_C", []) or []
    per_frame = [
        np.asarray(T, dtype=np.float64).reshape(4, 4) for T in per_frame_raw
    ]
    return ExtrinsicsRecord(
        W_T_C=W_T_C,
        n_frames=int(raw["n_frames"]),
        reprojection_error_px_mean=float(raw["reprojection_error_px_mean"]),
        reprojection_error_px_max=float(raw["reprojection_error_px_max"]),
        board=board,
        target=target,
        intrinsics_path=str(raw.get("intrinsics_path", "")),
        detection_mode=str(raw.get("detection_mode", "charuco")),
        created_at=str(raw.get("created_at", "")),
        notes=str(raw.get("notes", "")),
        camera=raw.get("camera"),
        per_frame_W_T_C=per_frame,
    )


# ----- board config -----


def load_board(path: str | Path) -> CharucoConfig:
    """Load a board YAML into :class:`CharucoConfig`."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return CharucoConfig(
        squares_x=int(raw["squares_x"]),
        squares_y=int(raw["squares_y"]),
        square_size_m=float(raw["square_size_m"]),
        marker_size_m=float(raw["marker_size_m"]),
        dictionary=str(raw.get("dictionary", "DICT_5X5_100")),
    )


def load_target_descriptor(path: str | Path) -> dict[str, Any]:
    """Load a target descriptor YAML.

    Supported descriptor shapes:

    ``type: charuco_board`` with the same fields as ``board.yaml``.
    ``type: aruco_marker`` with ``dictionary``, ``marker_id``, and ``size_m``.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: target descriptor must be a mapping")
    target_type = str(raw.get("type", "charuco_board"))
    if target_type == "charuco_board":
        board = CharucoConfig(
            squares_x=int(raw["squares_x"]),
            squares_y=int(raw["squares_y"]),
            square_size_m=float(raw["square_size_m"]),
            marker_size_m=float(raw["marker_size_m"]),
            dictionary=str(raw.get("dictionary", "DICT_5X5_100")),
        )
        return _target_from_board(board)
    if target_type == "aruco_marker":
        size_m = float(raw["size_m"])
        if size_m <= 0:
            raise ValueError(f"{path}: size_m must be positive, got {size_m}")
        marker_id = int(raw["marker_id"])
        if marker_id < 0:
            raise ValueError(f"{path}: marker_id must be non-negative, got {marker_id}")
        return {
            "type": "aruco_marker",
            "dictionary": str(raw.get("dictionary", "DICT_5X5_100")),
            "marker_id": marker_id,
            "size_m": size_m,
            "origin": str(raw.get("origin", "center")),
        }
    raise ValueError(f"{path}: unsupported target type {target_type!r}")


# ----- helpers -----


def _matrix_to_list(M: np.ndarray, *, expected_shape: tuple[int, int]) -> list[list[float]]:
    arr = np.asarray(M, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(
            f"matrix must have shape {expected_shape}, got {arr.shape}"
        )
    return [[float(x) for x in row] for row in arr]


def _board_to_dict(board: CharucoConfig) -> dict[str, Any]:
    return {
        "squares_x": int(board.squares_x),
        "squares_y": int(board.squares_y),
        "square_size_m": float(board.square_size_m),
        "marker_size_m": float(board.marker_size_m),
        "dictionary": str(board.dictionary),
    }


def _target_from_board(board: CharucoConfig | None) -> dict[str, Any]:
    if board is None:
        raise ValueError("ExtrinsicsRecord requires either target or board")
    return {"type": "charuco_board", **_board_to_dict(board)}


def _board_from_extrinsics_raw(raw: dict[str, Any]) -> CharucoConfig | None:
    board_raw = raw.get("board")
    if board_raw is None:
        target = raw.get("target") or {}
        if target.get("type") != "charuco_board":
            return None
        board_raw = target
    return CharucoConfig(
        squares_x=int(board_raw["squares_x"]),
        squares_y=int(board_raw["squares_y"]),
        square_size_m=float(board_raw["square_size_m"]),
        marker_size_m=float(board_raw["marker_size_m"]),
        dictionary=str(board_raw.get("dictionary", "DICT_5X5_100")),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ExtrinsicsRecord",
    "IntrinsicsRecord",
    "RealSenseIntrinsicsBundle",
    "RealSenseIntrinsicsCameraRecord",
    "load_board",
    "load_extrinsics",
    "load_intrinsics",
    "load_realsense_intrinsics_bundle",
    "load_target_descriptor",
    "profile_image_size",
    "resolve_intrinsics_file",
    "save_extrinsics",
    "save_intrinsics",
]
