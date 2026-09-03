from __future__ import annotations

import cv2
import numpy as np
import pytest

from camcal.aruco import ArucoMultiMarkerDetector
from camcal.charuco import aruco_dictionary
from camcal.marker_map import MarkerManifest, MarkerSpec


def test_multi_marker_detector_returns_all_configured_visible_ids_and_sizes():
    manifest = MarkerManifest(
        dictionary="DICT_6X6_250",
        markers=(
            MarkerSpec(2, nominal_size_m=0.20, measured_size_m=0.198),
            MarkerSpec(17, nominal_size_m=0.24),
        ),
    )
    detector = ArucoMultiMarkerDetector(manifest)
    dictionary = aruco_dictionary(manifest.dictionary)
    image = np.full((420, 780), 255, dtype=np.uint8)
    image[60:300, 40:280] = cv2.aruco.generateImageMarker(dictionary, 17, 240)
    image[80:300, 320:540] = cv2.aruco.generateImageMarker(dictionary, 2, 220)
    image[80:300, 560:780] = cv2.aruco.generateImageMarker(dictionary, 99, 220)

    detections = detector.detect(image)

    assert [detection.marker_id for detection in detections] == [2, 17]
    assert all(detection.image_points.shape == (4, 2) for detection in detections)
    assert all(detection.object_points.shape == (4, 3) for detection in detections)
    side_by_id = {
        detection.marker_id: np.linalg.norm(
            detection.object_points[1] - detection.object_points[0]
        )
        for detection in detections
    }
    assert side_by_id[2] == pytest.approx(0.198)
    assert side_by_id[17] == pytest.approx(0.24)


def test_multi_marker_detector_has_no_fixed_marker_count():
    manifest = MarkerManifest(
        dictionary="DICT_6X6_250",
        markers=tuple(MarkerSpec(marker_id, 0.10) for marker_id in (1, 8, 31)),
    )
    detector = ArucoMultiMarkerDetector(manifest)

    assert detector.detect(np.full((120, 160), 255, dtype=np.uint8)) == ()
