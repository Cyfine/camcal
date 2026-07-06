from argparse import Namespace

import pytest

from camcal.calibrate_extrinsics import (
    _looks_like_realsense,
    _resolve_aruco_size_m,
    _resolve_live_capture_settings,
    _resolve_target,
)
from camcal.list_cameras import CameraInfo


def test_resolve_live_capture_settings_defaults_to_factory_profile():
    args = Namespace(width=None, height=None, fps=None)
    width, height, fps = _resolve_live_capture_settings(
        args=args,
        intrinsics_profile={"fps": 30, "image_size": [1920, 1080]},
    )
    assert (width, height, fps) == (1920, 1080, 30)


def test_resolve_live_capture_settings_rejects_factory_profile_mismatch():
    args = Namespace(width=1280, height=720, fps=30)
    with pytest.raises(ValueError):
        _resolve_live_capture_settings(
            args=args,
            intrinsics_profile={"fps": 30, "image_size": [1920, 1080]},
        )


def test_looks_like_realsense_uses_vendor_or_model():
    assert _looks_like_realsense(
        CameraInfo(
            by_id_path=None,
            dev_path="/dev/video0",
            cv2_index=0,
            model="Intel RealSense D415",
            vendor="Intel",
            serial="840412062035",
        )
    )
    assert not _looks_like_realsense(
        CameraInfo(
            by_id_path=None,
            dev_path="/dev/video1",
            cv2_index=1,
            model="HD Pro Webcam C920",
            vendor="Logitech",
            serial="A1B2",
        )
    )


def test_resolve_aruco_size_m_from_mm():
    args = Namespace(aruco_size_m=None, aruco_size_mm=197.5)
    assert _resolve_aruco_size_m(args) == pytest.approx(0.1975)


def test_resolve_standalone_aruco_target_from_descriptor(tmp_path):
    descriptor = tmp_path / "aruco.yaml"
    descriptor.write_text(
        "type: aruco_marker\n"
        "dictionary: DICT_5X5_100\n"
        "marker_id: 7\n"
        "size_m: 0.1975\n"
    )
    target = _resolve_target(
        Namespace(
            target_descriptor=descriptor,
            detector=None,
            aruco_marker_id=None,
            aruco_size_m=None,
            aruco_size_mm=None,
            aruco_dictionary="DICT_5X5_100",
            board=None,
        )
    )
    assert target.detection_mode == "aruco_marker"
    assert target.board is None
    assert target.target["type"] == "aruco_marker"
    assert target.target["size_m"] == pytest.approx(0.1975)


def test_detector_aruco_requires_marker_descriptor_or_options():
    with pytest.raises(ValueError, match="--aruco-marker-id"):
        _resolve_target(
            Namespace(
                target_descriptor=None,
                detector="aruco",
                aruco_marker_id=None,
                aruco_size_m=None,
                aruco_size_mm=None,
                aruco_dictionary="DICT_5X5_100",
                board=None,
            )
        )
