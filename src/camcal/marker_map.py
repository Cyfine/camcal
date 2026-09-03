"""Marker inventory for a fixed multi-camera calibration workspace.

The printed black square supplies metric scale.  ``nominal_size_m`` is the
requested print size; ``measured_size_m`` is the operator's ruler/caliper
measurement after printing.  Calibration uses the measured value when it is
available and otherwise makes the nominal fallback explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import yaml

from .charuco import aruco_dictionary


DEFAULT_DICTIONARY = "DICT_6X6_250"
DEFAULT_MARKER_IDS = tuple(range(10))
DEFAULT_NOMINAL_SIZE_M = 0.240


@dataclass(frozen=True)
class MarkerSpec:
    """One physical marker, identified by ID within one ArUco dictionary."""

    marker_id: int
    nominal_size_m: float
    measured_size_m: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.marker_id, bool) or not isinstance(self.marker_id, int):
            raise TypeError("marker_id must be an integer")
        if self.marker_id < 0:
            raise ValueError(f"marker_id must be non-negative, got {self.marker_id}")
        _require_positive_finite(self.nominal_size_m, "nominal_size_m")
        if self.measured_size_m is not None:
            _require_positive_finite(self.measured_size_m, "measured_size_m")

    @property
    def side_length_m(self) -> float:
        """Physical black-square side used by geometry, in metres."""
        return float(
            self.measured_size_m
            if self.measured_size_m is not None
            else self.nominal_size_m
        )

    @property
    def size_m(self) -> float:
        """Compatibility alias for :attr:`side_length_m`."""
        return self.side_length_m

    @property
    def uses_measured_size(self) -> bool:
        return self.measured_size_m is not None


@dataclass(frozen=True)
class MarkerManifest:
    """The unique marker IDs and physical sizes installed in one workspace."""

    dictionary: str
    markers: tuple[MarkerSpec, ...]

    def __post_init__(self) -> None:
        dictionary = aruco_dictionary(self.dictionary)
        if not self.markers:
            raise ValueError("markers must not be empty")
        ids = [marker.marker_id for marker in self.markers]
        if len(set(ids)) != len(ids):
            duplicates = sorted({marker_id for marker_id in ids if ids.count(marker_id) > 1})
            raise ValueError(f"marker IDs must be unique, duplicates: {duplicates}")
        capacity = int(dictionary.bytesList.shape[0])
        invalid = sorted(marker_id for marker_id in ids if marker_id >= capacity)
        if invalid:
            raise ValueError(
                f"marker IDs {invalid} exceed {self.dictionary} range 0..{capacity - 1}"
            )

    @property
    def by_id(self) -> dict[int, MarkerSpec]:
        return {marker.marker_id: marker for marker in self.markers}

    def get(self, marker_id: int) -> MarkerSpec:
        """Return the configured marker or raise ``KeyError``."""
        try:
            return self.by_id[int(marker_id)]
        except KeyError:
            raise KeyError(f"marker ID {marker_id} is not in the manifest") from None


def default_marker_manifest(
    marker_ids: Iterable[int] = DEFAULT_MARKER_IDS,
    *,
    nominal_size_m: float = DEFAULT_NOMINAL_SIZE_M,
    dictionary: str = DEFAULT_DICTIONARY,
) -> MarkerManifest:
    """Create the default printable inventory (IDs 0--9 unless overridden)."""
    return MarkerManifest(
        dictionary=dictionary,
        markers=tuple(
            MarkerSpec(marker_id=int(marker_id), nominal_size_m=nominal_size_m)
            for marker_id in marker_ids
        ),
    )


def save_marker_manifest(path: str | Path, manifest: MarkerManifest) -> None:
    """Write a compact, human-editable YAML marker inventory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "dictionary": manifest.dictionary,
        "markers": [
            {
                "id": marker.marker_id,
                "nominal_black_square_side_m": marker.nominal_size_m,
                "measured_black_square_side_m": marker.measured_size_m,
            }
            for marker in manifest.markers
        ],
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def load_marker_manifest(path: str | Path) -> MarkerManifest:
    """Read and validate a marker inventory written by :func:`save_marker_manifest`."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: marker manifest must be a YAML mapping")
    raw_markers = raw.get("markers")
    if not isinstance(raw_markers, list):
        raise ValueError(f"{path}: markers must be a YAML list")
    try:
        markers = tuple(
            MarkerSpec(
                marker_id=entry["id"],
                nominal_size_m=entry["nominal_black_square_side_m"],
                measured_size_m=entry.get("measured_black_square_side_m"),
            )
            for entry in raw_markers
        )
        return MarkerManifest(dictionary=str(raw["dictionary"]), markers=markers)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid marker entry: {exc}") from exc


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


__all__ = [
    "DEFAULT_DICTIONARY",
    "DEFAULT_MARKER_IDS",
    "DEFAULT_NOMINAL_SIZE_M",
    "MarkerManifest",
    "MarkerSpec",
    "default_marker_manifest",
    "load_marker_manifest",
    "save_marker_manifest",
]
