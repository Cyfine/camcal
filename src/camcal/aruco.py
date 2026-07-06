"""Standalone ArUco marker target detection for extrinsic calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .charuco import aruco_dictionary


@dataclass(frozen=True)
class ArucoMarkerConfig:
    """A single square ArUco marker used as the world-frame target.

    The marker frame origin is at the marker center, with +X to the marker's
    right, +Y toward the marker top edge, and Z=0 on the paper plane.
    Object points follow OpenCV's ``SOLVEPNP_IPPE_SQUARE`` order:
    top-left, top-right, bottom-right, bottom-left.
    """

    dictionary: str
    marker_id: int
    size_m: float

    def __post_init__(self) -> None:
        aruco_dictionary(self.dictionary)
        if self.marker_id < 0:
            raise ValueError(f"marker_id must be non-negative, got {self.marker_id}")
        if self.size_m <= 0:
            raise ValueError(f"size_m must be positive, got {self.size_m}")

    def to_target_dict(self) -> dict[str, object]:
        return {
            "type": "aruco_marker",
            "dictionary": self.dictionary,
            "marker_id": int(self.marker_id),
            "size_m": float(self.size_m),
            "origin": "center",
        }


@dataclass
class ArucoMarkerDetection:
    """A successful detection of the configured standalone ArUco marker."""

    object_points: np.ndarray
    image_points: np.ndarray
    marker_id: int


class ArucoMarkerTarget:
    """Detector and geometry for one standalone ArUco marker."""

    def __init__(self, config: ArucoMarkerConfig) -> None:
        self.config = config
        self.detector = cv2.aruco.ArucoDetector(aruco_dictionary(config.dictionary))
        half = float(config.size_m) / 2.0
        self.object_points = np.asarray(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    def detect(self, image: np.ndarray) -> ArucoMarkerDetection | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        if marker_ids is None or not marker_corners:
            return None

        ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        for corners, marker_id in zip(marker_corners, ids, strict=True):
            if int(marker_id) != self.config.marker_id:
                continue
            image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            return ArucoMarkerDetection(
                object_points=self.object_points.copy(),
                image_points=image_points,
                marker_id=int(marker_id),
            )
        return None


__all__ = ["ArucoMarkerConfig", "ArucoMarkerDetection", "ArucoMarkerTarget"]
