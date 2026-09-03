"""Portable static camera/marker corner observations.

The workspace cameras and printed markers are fixed during calibration, so
captures made at different times or on different hosts may be merged.  Each
camera snapshot retains the exact projection model used to interpret its
pixels; each marker observation retains the source image and capture identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .fleet import FleetCamera, load_fleet
from .io_yaml import resolve_intrinsics_file


@dataclass(frozen=True)
class CameraSnapshot:
    """One camera's fixed intrinsics and image profile for an observation set."""

    name: str
    K: np.ndarray
    d: np.ndarray
    image_size: tuple[int, int]
    intrinsics_ref: str = ""
    intrinsics_source: str = ""
    profile: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        d = np.asarray(self.d, dtype=np.float64).reshape(-1)
        if not self.name.strip():
            raise ValueError("camera name must be non-empty")
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"camera {self.name!r}: K must be finite 3x3")
        if d.size == 0 or not np.all(np.isfinite(d)):
            raise ValueError(
                f"camera {self.name!r}: distortion coefficients must be finite"
            )
        if len(self.image_size) != 2 or any(int(v) <= 0 for v in self.image_size):
            raise ValueError(
                f"camera {self.name!r}: image_size must be positive [width, height]"
            )
        object.__setattr__(self, "K", K.copy())
        object.__setattr__(self, "d", d.copy())
        object.__setattr__(
            self, "image_size", (int(self.image_size[0]), int(self.image_size[1]))
        )


@dataclass(frozen=True)
class MarkerObservation:
    """Four ordered pixel corners of one marker in one captured camera image."""

    camera_name: str
    capture_id: str
    marker_id: int
    corners_px: np.ndarray
    source_image: str = ""
    captured_at: str = ""
    source_host: str = ""

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_px, dtype=np.float64)
        if not self.camera_name.strip():
            raise ValueError("observation camera_name must be non-empty")
        if not self.capture_id.strip():
            raise ValueError("observation capture_id must be non-empty")
        if isinstance(self.marker_id, bool) or int(self.marker_id) < 0:
            raise ValueError(f"marker_id must be non-negative, got {self.marker_id}")
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("corners_px must be a finite 4x2 array")
        object.__setattr__(self, "marker_id", int(self.marker_id))
        object.__setattr__(self, "corners_px", corners.copy())


@dataclass(frozen=True)
class ObservationDataset:
    """Static observations from one or more cameras and hosts."""

    dictionary: str
    cameras: tuple[CameraSnapshot, ...]
    observations: tuple[MarkerObservation, ...]
    created_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.dictionary.strip():
            raise ValueError("dictionary must be non-empty")
        names = [camera.name for camera in self.cameras]
        if not names:
            raise ValueError("observation dataset must contain at least one camera")
        if len(names) != len(set(names)):
            raise ValueError("camera names must be unique within one observation set")
        known = set(names)
        unknown = sorted(
            {obs.camera_name for obs in self.observations if obs.camera_name not in known}
        )
        if unknown:
            raise ValueError(f"observations reference unknown cameras: {unknown}")

    @property
    def camera_by_name(self) -> dict[str, CameraSnapshot]:
        return {camera.name: camera for camera in self.cameras}


def camera_snapshot_from_fleet(
    camera: FleetCamera,
    *,
    fleet_directory: str | Path,
) -> CameraSnapshot:
    """Resolve one fleet camera into a self-contained intrinsics snapshot."""
    if camera.intrinsics is None:
        raise ValueError(f"fleet camera {camera.name!r} has no intrinsics")
    intr = camera.intrinsics
    if intr.K is not None and intr.d is not None and intr.image_size is not None:
        K = intr.K
        d = intr.d
        image_size = intr.image_size
        source = intr.source
    elif intr.path:
        path = Path(intr.path)
        if not path.is_absolute():
            path = Path(fleet_directory) / path
        serial = str(camera.device.get("serial", "")) or None
        record, resolved_profile = resolve_intrinsics_file(
            path, camera_serial=serial
        )
        K, d, image_size = record.K, record.d, record.image_size
        source = record.source
        if intr.profile is None and resolved_profile is not None:
            intr_profile = dict(resolved_profile)
        else:
            intr_profile = intr.profile
        return CameraSnapshot(
            name=camera.name,
            K=K,
            d=d,
            image_size=image_size,
            intrinsics_ref=str(path),
            intrinsics_source=source,
            profile=intr_profile,
        )
    else:
        raise ValueError(
            f"fleet camera {camera.name!r} has neither inline intrinsics nor a path"
        )
    return CameraSnapshot(
        name=camera.name,
        K=K,
        d=d,
        image_size=image_size,
        intrinsics_ref=intr.path,
        intrinsics_source=source,
        profile=intr.profile,
    )


def camera_snapshots_from_fleet(
    fleet_path: str | Path,
    *,
    camera_names: Iterable[str] | None = None,
) -> tuple[CameraSnapshot, ...]:
    """Resolve enabled fleet cameras, optionally restricted by name."""
    fleet_path = Path(fleet_path)
    fleet = load_fleet(fleet_path)
    selected = set(camera_names) if camera_names is not None else None
    cameras = [
        camera
        for camera in fleet.cameras
        if camera.enabled and (selected is None or camera.name in selected)
    ]
    if selected is not None:
        missing = sorted(selected - {camera.name for camera in cameras})
        if missing:
            raise ValueError(f"fleet has no enabled cameras named {missing}")
    if not cameras:
        raise ValueError("no enabled fleet cameras selected")
    return tuple(
        camera_snapshot_from_fleet(camera, fleet_directory=fleet_path.parent)
        for camera in cameras
    )


def merge_observation_datasets(
    datasets: Iterable[ObservationDataset],
) -> ObservationDataset:
    """Merge static observation sets while rejecting camera-model conflicts."""
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("at least one observation dataset is required")
    dictionary = datasets[0].dictionary
    cameras: dict[str, CameraSnapshot] = {}
    observations: list[MarkerObservation] = []
    for dataset in datasets:
        if dataset.dictionary != dictionary:
            raise ValueError(
                f"cannot merge dictionaries {dictionary!r} and {dataset.dictionary!r}"
            )
        for camera in dataset.cameras:
            previous = cameras.get(camera.name)
            if previous is not None and not _same_camera_model(previous, camera):
                raise ValueError(
                    f"camera {camera.name!r} has conflicting intrinsics/profile"
                )
            cameras[camera.name] = camera
        observations.extend(dataset.observations)
    return ObservationDataset(
        dictionary=dictionary,
        cameras=tuple(cameras[name] for name in sorted(cameras)),
        observations=tuple(observations),
        created_at=_utc_now_iso(),
        notes=f"merged {len(datasets)} static observation set(s)",
    )


def save_observation_dataset(
    path: str | Path,
    dataset: ObservationDataset,
) -> None:
    """Write a portable, human-readable observation YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "version": 1,
        "dictionary": dataset.dictionary,
        "created_at": dataset.created_at or _utc_now_iso(),
        "cameras": [
            {
                "name": camera.name,
                "image_size": list(camera.image_size),
                "K": camera.K.tolist(),
                "d": camera.d.tolist(),
                "intrinsics_ref": camera.intrinsics_ref,
                "intrinsics_source": camera.intrinsics_source,
                "profile": camera.profile,
            }
            for camera in dataset.cameras
        ],
        "observations": [
            {
                "camera": observation.camera_name,
                "capture_id": observation.capture_id,
                "marker_id": observation.marker_id,
                "corners_px": observation.corners_px.tolist(),
                "source_image": observation.source_image,
                "captured_at": observation.captured_at,
                "source_host": observation.source_host,
            }
            for observation in dataset.observations
        ],
    }
    if dataset.notes:
        body["notes"] = dataset.notes
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_observation_dataset(path: str | Path) -> ObservationDataset:
    """Load an observation YAML written by :func:`save_observation_dataset`."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
        raise ValueError(f"{path}: unsupported observation schema")
    try:
        cameras = tuple(
            CameraSnapshot(
                name=str(item["name"]),
                K=np.asarray(item["K"], dtype=np.float64),
                d=np.asarray(item["d"], dtype=np.float64),
                image_size=tuple(item["image_size"]),
                intrinsics_ref=str(item.get("intrinsics_ref", "")),
                intrinsics_source=str(item.get("intrinsics_source", "")),
                profile=item.get("profile"),
            )
            for item in raw["cameras"]
        )
        observations = tuple(
            MarkerObservation(
                camera_name=str(item["camera"]),
                capture_id=str(item["capture_id"]),
                marker_id=int(item["marker_id"]),
                corners_px=np.asarray(item["corners_px"], dtype=np.float64),
                source_image=_resolve_source_image(
                    path.parent,
                    str(item.get("source_image", "")),
                ),
                captured_at=str(item.get("captured_at", "")),
                source_host=str(item.get("source_host", "")),
            )
            for item in raw.get("observations", [])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid observation data: {exc}") from exc
    return ObservationDataset(
        dictionary=str(raw["dictionary"]),
        cameras=cameras,
        observations=observations,
        created_at=str(raw.get("created_at", "")),
        notes=str(raw.get("notes", "")),
    )


def _same_camera_model(a: CameraSnapshot, b: CameraSnapshot) -> bool:
    return (
        a.image_size == b.image_size
        and a.K.shape == b.K.shape
        and a.d.shape == b.d.shape
        and np.allclose(a.K, b.K, rtol=0.0, atol=1e-12)
        and np.allclose(a.d, b.d, rtol=0.0, atol=1e-12)
    )


def _resolve_source_image(dataset_directory: Path, source_image: str) -> str:
    if not source_image:
        return ""
    source_path = Path(source_image)
    if not source_path.is_absolute():
        source_path = dataset_directory / source_path
    return str(source_path.resolve())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "CameraSnapshot",
    "MarkerObservation",
    "ObservationDataset",
    "camera_snapshot_from_fleet",
    "camera_snapshots_from_fleet",
    "load_observation_dataset",
    "merge_observation_datasets",
    "save_observation_dataset",
]
