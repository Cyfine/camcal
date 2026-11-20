"""ChArUco corner observations of one board that moves between captures.

The cameras stay fixed while the board is carried through the shared field of
view. Every capture is one static board pose, seen by any subset of the
cameras at the same instant; its ``capture_id`` is therefore shared by all the
cameras that saw it. This is the opposite of :mod:`camcal.observations`,
where the markers are fixed and each camera image is its own capture.

Each capture has a role:

``solve``
    Used by the bundle adjustment.
``world``
    Used by the bundle adjustment, and its board frame defines the world.
``validation``
    Kept out of the bundle adjustment; used to check the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .charuco import CharucoBoard, CharucoConfig
from .io_yaml import _board_to_dict
from .observations import CameraSnapshot

CAPTURE_ROLES = ("solve", "world", "validation")


@dataclass(frozen=True)
class BoardCapture:
    """One static board pose."""

    capture_id: str
    role: str = "solve"
    captured_at: str = ""

    def __post_init__(self) -> None:
        if not self.capture_id.strip():
            raise ValueError("capture_id must be non-empty")
        if self.role not in CAPTURE_ROLES:
            raise ValueError(f"capture role must be one of {CAPTURE_ROLES}, got {self.role!r}")


@dataclass(frozen=True)
class BoardObservation:
    """The ChArUco corners one camera detected in one capture."""

    camera_name: str
    capture_id: str
    corner_ids: np.ndarray
    image_points: np.ndarray
    source_image: str = ""

    def __post_init__(self) -> None:
        ids = np.asarray(self.corner_ids, dtype=np.int64).reshape(-1)
        points = np.asarray(self.image_points, dtype=np.float64)
        if points.shape != (ids.size, 2) or not np.all(np.isfinite(points)):
            raise ValueError("image_points must be a finite (N, 2) array matching corner_ids")
        if ids.size == 0 or np.any(ids < 0) or len(set(ids.tolist())) != ids.size:
            raise ValueError("corner_ids must be unique non-negative indices")
        object.__setattr__(self, "corner_ids", ids.copy())
        object.__setattr__(self, "image_points", points.copy())


@dataclass(frozen=True)
class BoardObservationDataset:
    """Board, cameras with their prior intrinsics, captures and observations."""

    board: CharucoConfig
    cameras: tuple[CameraSnapshot, ...]
    captures: tuple[BoardCapture, ...]
    observations: tuple[BoardObservation, ...]
    created_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        names = [camera.name for camera in self.cameras]
        if not names or len(names) != len(set(names)):
            raise ValueError("camera names must be present and unique")
        capture_ids = [capture.capture_id for capture in self.captures]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("capture ids must be unique")
        if sum(capture.role == "world" for capture in self.captures) > 1:
            raise ValueError("at most one capture may define the world")
        corner_count = (self.board.squares_x - 1) * (self.board.squares_y - 1)
        seen = set()
        for observation in self.observations:
            if observation.camera_name not in names:
                raise ValueError(f"observation of unknown camera {observation.camera_name!r}")
            if observation.capture_id not in capture_ids:
                raise ValueError(f"observation of unknown capture {observation.capture_id!r}")
            key = (observation.camera_name, observation.capture_id)
            if key in seen:
                raise ValueError(f"duplicate observation {key}")
            seen.add(key)
            if np.any(observation.corner_ids >= corner_count):
                raise ValueError(f"corner id out of range in {key}")

    @property
    def world_capture(self) -> str | None:
        for capture in self.captures:
            if capture.role == "world":
                return capture.capture_id
        return None

    def capture_ids(self, *roles: str) -> list[str]:
        return [capture.capture_id for capture in self.captures if capture.role in roles]


def board_corner_points(board: CharucoConfig) -> np.ndarray:
    """Inner corners in the board frame, indexed by ChArUco corner id (z = 0).

    Taken from OpenCV's board so the ids are the ones its detector reports.
    """
    corners = CharucoBoard.from_config(board).cv_board.getChessboardCorners()
    return np.asarray(corners, dtype=np.float64).reshape(-1, 3)


def save_board_observations(path: str | Path, dataset: BoardObservationDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "version": 1,
        "kind": "charuco_moving_board",
        "created_at": dataset.created_at or _utc_now_iso(),
        "board": _board_to_dict(dataset.board),
        "cameras": [
            {
                "name": camera.name,
                "image_size": list(camera.image_size),
                "K": camera.K.tolist(),
                "d": camera.d.tolist(),
                "intrinsics_source": camera.intrinsics_source,
                "profile": camera.profile,
            }
            for camera in dataset.cameras
        ],
        "captures": [
            {"id": capture.capture_id, "role": capture.role, "captured_at": capture.captured_at}
            for capture in dataset.captures
        ],
        "observations": [
            {
                "camera": observation.camera_name,
                "capture_id": observation.capture_id,
                "corner_ids": observation.corner_ids.tolist(),
                "image_points": observation.image_points.tolist(),
                "source_image": observation.source_image,
            }
            for observation in dataset.observations
        ],
    }
    if dataset.notes:
        body["notes"] = dataset.notes
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_board_observations(path: str | Path) -> BoardObservationDataset:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("kind") != "charuco_moving_board":
        raise ValueError(f"{path}: not a moving-board observation file")
    if int(raw.get("version", 0)) != 1:
        raise ValueError(f"{path}: unsupported version")
    board_raw = raw["board"]
    board = CharucoConfig(
        squares_x=int(board_raw["squares_x"]),
        squares_y=int(board_raw["squares_y"]),
        square_size_m=float(board_raw["square_size_m"]),
        marker_size_m=float(board_raw["marker_size_m"]),
        dictionary=str(board_raw["dictionary"]),
        legacy_pattern=bool(board_raw.get("legacy_pattern", False)),
    )
    cameras = tuple(
        CameraSnapshot(
            name=str(item["name"]),
            K=np.asarray(item["K"], dtype=np.float64),
            d=np.asarray(item["d"], dtype=np.float64),
            image_size=tuple(item["image_size"]),
            intrinsics_source=str(item.get("intrinsics_source", "")),
            profile=item.get("profile"),
        )
        for item in raw["cameras"]
    )
    captures = tuple(
        BoardCapture(str(item["id"]), str(item.get("role", "solve")), str(item.get("captured_at", "")))
        for item in raw.get("captures", [])
    )
    observations = tuple(
        BoardObservation(
            camera_name=str(item["camera"]),
            capture_id=str(item["capture_id"]),
            corner_ids=np.asarray(item["corner_ids"], dtype=np.int64),
            image_points=np.asarray(item["image_points"], dtype=np.float64),
            source_image=str(item.get("source_image", "")),
        )
        for item in raw.get("observations", [])
    )
    return BoardObservationDataset(
        board=board,
        cameras=cameras,
        captures=captures,
        observations=observations,
        created_at=str(raw.get("created_at", "")),
        notes=str(raw.get("notes", "")),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "CAPTURE_ROLES",
    "BoardCapture",
    "BoardObservation",
    "BoardObservationDataset",
    "board_corner_points",
    "load_board_observations",
    "save_board_observations",
]
