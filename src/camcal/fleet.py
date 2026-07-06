"""Stateful rig/fleet configuration for multi-camera systems.

Unlike the single-camera intrinsics/extrinsics artifacts, a fleet config is a
long-lived operational record: named cameras, stable device identity, and the
current calibration state that places every camera in a shared world frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import se3
from .io_yaml import ExtrinsicsRecord, IntrinsicsRecord


@dataclass
class FleetIntrinsics:
    """Named-camera intrinsics state stored inside a fleet config."""

    path: str = ""
    source: str = ""
    image_size: tuple[int, int] | None = None
    K: np.ndarray | None = None
    d: np.ndarray | None = None
    camera: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    notes: str = ""

    @classmethod
    def from_record(
        cls,
        record: IntrinsicsRecord,
        *,
        path: str = "",
        profile: dict[str, Any] | None = None,
    ) -> FleetIntrinsics:
        return cls(
            path=path,
            source=record.source,
            image_size=record.image_size,
            K=np.asarray(record.K, dtype=np.float64).copy(),
            d=np.asarray(record.d, dtype=np.float64).copy(),
            camera=record.camera,
            profile=profile,
            notes=record.notes,
        )


@dataclass
class FleetExtrinsics:
    """Named-camera extrinsics state stored inside a fleet config."""

    path: str = ""
    detection_mode: str = "charuco"
    W_T_C: np.ndarray | None = None
    reprojection_error_px_mean: float | None = None
    reprojection_error_px_max: float | None = None
    camera: dict[str, Any] | None = None
    notes: str = ""

    @classmethod
    def from_record(
        cls,
        record: ExtrinsicsRecord,
        *,
        path: str = "",
    ) -> FleetExtrinsics:
        return cls(
            path=path,
            detection_mode=record.detection_mode,
            W_T_C=np.asarray(record.W_T_C, dtype=np.float64).copy(),
            reprojection_error_px_mean=record.reprojection_error_px_mean,
            reprojection_error_px_max=record.reprojection_error_px_max,
            camera=record.camera,
            notes=record.notes,
        )


@dataclass
class FleetCamera:
    """One named camera in a multi-camera rig."""

    name: str
    enabled: bool = True
    device: dict[str, Any] = field(default_factory=dict)
    intrinsics: FleetIntrinsics | None = None
    extrinsics: FleetExtrinsics | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FleetCamera.name must be non-empty")


@dataclass
class CameraFleetRecord:
    """Stateful multi-camera rig configuration."""

    name: str
    cameras: list[FleetCamera] = field(default_factory=list)
    board_path: str = ""
    world_frame: str = "world"
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CameraFleetRecord.name must be non-empty")

    def get_camera(self, name: str) -> FleetCamera:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"Unknown fleet camera {name!r}")

    def upsert_camera(self, camera: FleetCamera) -> None:
        for index, existing in enumerate(self.cameras):
            if existing.name == camera.name:
                self.cameras[index] = camera
                return
        self.cameras.append(camera)


def save_fleet(path: str | Path, record: CameraFleetRecord) -> None:
    """Write a stateful fleet config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "version": 1,
        "name": record.name,
        "world_frame": record.world_frame,
        "created_at": record.created_at or _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "cameras": [_camera_to_dict(camera) for camera in record.cameras],
    }
    if record.board_path:
        body["board_path"] = record.board_path
    if record.notes:
        body["notes"] = record.notes
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_fleet(path: str | Path) -> CameraFleetRecord:
    """Read a stateful fleet config from YAML."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cameras = [_camera_from_dict(camera_raw) for camera_raw in raw.get("cameras", [])]
    return CameraFleetRecord(
        name=str(raw["name"]),
        cameras=cameras,
        board_path=str(raw.get("board_path", "")),
        world_frame=str(raw.get("world_frame", "world")),
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        notes=str(raw.get("notes", "")),
    )


def _camera_to_dict(camera: FleetCamera) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": camera.name,
        "enabled": bool(camera.enabled),
    }
    if camera.device:
        body["device"] = camera.device
    if camera.intrinsics is not None:
        body["intrinsics"] = _intrinsics_to_dict(camera.intrinsics)
    if camera.extrinsics is not None:
        body["extrinsics"] = _extrinsics_to_dict(camera.extrinsics)
    if camera.notes:
        body["notes"] = camera.notes
    return body


def _camera_from_dict(raw: dict[str, Any]) -> FleetCamera:
    return FleetCamera(
        name=str(raw["name"]),
        enabled=bool(raw.get("enabled", True)),
        device=dict(raw.get("device") or {}),
        intrinsics=_intrinsics_from_dict(raw["intrinsics"]) if "intrinsics" in raw else None,
        extrinsics=_extrinsics_from_dict(raw["extrinsics"]) if "extrinsics" in raw else None,
        notes=str(raw.get("notes", "")),
    )


def _intrinsics_to_dict(intrinsics: FleetIntrinsics) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source": intrinsics.source,
    }
    if intrinsics.path:
        body["path"] = intrinsics.path
    if intrinsics.image_size is not None:
        body["image_size"] = [int(intrinsics.image_size[0]), int(intrinsics.image_size[1])]
    if intrinsics.K is not None:
        body["K"] = _matrix_to_list(intrinsics.K, expected_shape=(3, 3))
    if intrinsics.d is not None:
        body["d"] = [float(x) for x in np.asarray(intrinsics.d).flatten()]
    if intrinsics.camera is not None:
        body["camera"] = intrinsics.camera
    if intrinsics.profile is not None:
        body["profile"] = intrinsics.profile
    if intrinsics.notes:
        body["notes"] = intrinsics.notes
    return body


def _intrinsics_from_dict(raw: dict[str, Any]) -> FleetIntrinsics:
    image_size = raw.get("image_size")
    return FleetIntrinsics(
        path=str(raw.get("path", "")),
        source=str(raw.get("source", "")),
        image_size=(
            tuple(int(x) for x in image_size)
            if image_size is not None
            else None
        ),
        K=np.asarray(raw["K"], dtype=np.float64) if "K" in raw else None,
        d=np.asarray(raw["d"], dtype=np.float64).flatten() if "d" in raw else None,
        camera=raw.get("camera"),
        profile=raw.get("profile"),
        notes=str(raw.get("notes", "")),
    )


def _extrinsics_to_dict(extrinsics: FleetExtrinsics) -> dict[str, Any]:
    body: dict[str, Any] = {
        "detection_mode": extrinsics.detection_mode,
    }
    if extrinsics.path:
        body["path"] = extrinsics.path
    if extrinsics.W_T_C is not None:
        se3.assert_se3(np.asarray(extrinsics.W_T_C, dtype=np.float64))
        quat_xyzw, t = se3.to_quat_t(np.asarray(extrinsics.W_T_C, dtype=np.float64))
        body["W_T_C"] = _matrix_to_list(extrinsics.W_T_C, expected_shape=(4, 4))
        body["translation_m"] = [float(x) for x in t]
        body["quaternion_xyzw"] = [float(x) for x in quat_xyzw]
    if extrinsics.reprojection_error_px_mean is not None:
        body["reprojection_error_px_mean"] = float(extrinsics.reprojection_error_px_mean)
    if extrinsics.reprojection_error_px_max is not None:
        body["reprojection_error_px_max"] = float(extrinsics.reprojection_error_px_max)
    if extrinsics.camera is not None:
        body["camera"] = extrinsics.camera
    if extrinsics.notes:
        body["notes"] = extrinsics.notes
    return body


def _extrinsics_from_dict(raw: dict[str, Any]) -> FleetExtrinsics:
    W_T_C = np.asarray(raw["W_T_C"], dtype=np.float64) if "W_T_C" in raw else None
    if W_T_C is not None:
        se3.assert_se3(W_T_C)
    return FleetExtrinsics(
        path=str(raw.get("path", "")),
        detection_mode=str(raw.get("detection_mode", "charuco")),
        W_T_C=W_T_C,
        reprojection_error_px_mean=(
            float(raw["reprojection_error_px_mean"])
            if "reprojection_error_px_mean" in raw
            else None
        ),
        reprojection_error_px_max=(
            float(raw["reprojection_error_px_max"])
            if "reprojection_error_px_max" in raw
            else None
        ),
        camera=raw.get("camera"),
        notes=str(raw.get("notes", "")),
    )


def _matrix_to_list(
    M: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> list[list[float]]:
    arr = np.asarray(M, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(
            f"matrix must have shape {expected_shape}, got {arr.shape}"
        )
    return [[float(x) for x in row] for row in arr]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "CameraFleetRecord",
    "FleetCamera",
    "FleetExtrinsics",
    "FleetIntrinsics",
    "load_fleet",
    "save_fleet",
]
