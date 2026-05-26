"""Flat, human-editable YAML schemas for intrinsics and extrinsics.

Two record types, each with a ``save_*`` / ``load_*`` pair:

* ``intrinsics.yaml`` — :class:`IntrinsicsRecord` (``K``, ``d``,
  ``image_size``, plus provenance).
* ``extrinsics.yaml`` — :class:`ExtrinsicsRecord` (``W_T_C`` plus the
  board snapshot the world frame is pinned to).

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


# ----- extrinsics -----


@dataclass
class ExtrinsicsRecord:
    """In-memory mirror of ``extrinsics.yaml``.

    ``W_T_C`` is the pose of the camera in the world frame. The world
    frame is defined by the ChArUco board snapshot in ``board``.
    """

    W_T_C: np.ndarray                   # (4, 4) SE(3)
    board: CharucoConfig
    n_frames: int
    reprojection_error_px_mean: float
    reprojection_error_px_max: float
    intrinsics_path: str = ""
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
    body: dict[str, Any] = {
        "W_T_C": _matrix_to_list(record.W_T_C, expected_shape=(4, 4)),
        "translation_m": [float(x) for x in t],
        "quaternion_xyzw": [float(x) for x in quat_xyzw],
        "n_frames": int(record.n_frames),
        "reprojection_error_px_mean": float(record.reprojection_error_px_mean),
        "reprojection_error_px_max": float(record.reprojection_error_px_max),
        "board": {
            "squares_x": int(record.board.squares_x),
            "squares_y": int(record.board.squares_y),
            "square_size_m": float(record.board.square_size_m),
            "marker_size_m": float(record.board.marker_size_m),
            "dictionary": str(record.board.dictionary),
        },
        "intrinsics_path": str(record.intrinsics_path),
        "created_at": record.created_at or _utc_now_iso(),
    }
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
    board_raw = raw["board"]
    board = CharucoConfig(
        squares_x=int(board_raw["squares_x"]),
        squares_y=int(board_raw["squares_y"]),
        square_size_m=float(board_raw["square_size_m"]),
        marker_size_m=float(board_raw["marker_size_m"]),
        dictionary=str(board_raw.get("dictionary", "DICT_5X5_100")),
    )
    per_frame_raw = raw.get("per_frame_W_T_C", []) or []
    per_frame = [
        np.asarray(T, dtype=np.float64).reshape(4, 4) for T in per_frame_raw
    ]
    return ExtrinsicsRecord(
        W_T_C=W_T_C,
        board=board,
        n_frames=int(raw["n_frames"]),
        reprojection_error_px_mean=float(raw["reprojection_error_px_mean"]),
        reprojection_error_px_max=float(raw["reprojection_error_px_max"]),
        intrinsics_path=str(raw.get("intrinsics_path", "")),
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


# ----- helpers -----


def _matrix_to_list(M: np.ndarray, *, expected_shape: tuple[int, int]) -> list[list[float]]:
    arr = np.asarray(M, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(
            f"matrix must have shape {expected_shape}, got {arr.shape}"
        )
    return [[float(x) for x in row] for row in arr]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ExtrinsicsRecord",
    "IntrinsicsRecord",
    "load_board",
    "load_extrinsics",
    "load_intrinsics",
    "save_extrinsics",
    "save_intrinsics",
]
