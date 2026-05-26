from camcal.list_cameras import (
    CameraInfo,
    ProbeResult,
    _dedup_by_usb,
    _format_human,
    _format_json,
    _format_paths,
    parse_by_id_name,
    vendor_name,
)


# ----- by-id filename parser -----


def test_parse_by_id_name_logitech_capture():
    parsed = parse_by_id_name(
        "usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0"
    )
    assert parsed is not None
    assert parsed.vid_or_vendor == "046d"
    assert parsed.product == "HD Pro Webcam C920"
    assert parsed.serial == "A1B2C3D4"
    assert parsed.subdevice_index == 0
    assert parsed.is_capture_interface


def test_parse_by_id_name_subdevice():
    parsed = parse_by_id_name(
        "usb-046d_HD_Pro_Webcam_C920_E5F6G7H8-video-index1"
    )
    assert parsed is not None
    assert parsed.subdevice_index == 1
    assert not parsed.is_capture_interface


def test_parse_by_id_name_realsense():
    parsed = parse_by_id_name(
        "usb-Intel_R__RealSense_TM__Depth_Camera_435i_123456789012-video-index0"
    )
    assert parsed is not None
    # Vendor lookup happens separately; here we just check the parser
    # accepts vendor-strings (not 4-hex VIDs).
    assert parsed.vid_or_vendor == "Intel"
    assert "RealSense" in parsed.product
    assert parsed.serial == "123456789012"


def test_parse_by_id_name_rejects_non_video_entries():
    assert parse_by_id_name("usb-046d_C920_ABCD-audio-controlC1") is None
    assert parse_by_id_name("not-a-v4l-by-id-name") is None


# ----- vendor lookup -----


def test_vendor_name_known_vid():
    assert vendor_name("046d") == "Logitech"
    assert vendor_name("046D") == "Logitech"  # case-insensitive
    assert vendor_name("8086") == "Intel"


def test_vendor_name_falls_back_to_raw():
    assert vendor_name("ffff") == "ffff"
    assert vendor_name("UnknownVendor") == "UnknownVendor"


# ----- formatters -----


def _sample_cameras() -> list[CameraInfo]:
    return [
        CameraInfo(
            by_id_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_AAAA-video-index0",
            dev_path="/dev/video0",
            cv2_index=0,
            model="HD Pro Webcam C920",
            vendor="Logitech",
            serial="AAAA",
        ),
        CameraInfo(
            by_id_path=None,                # built-in webcam without by-id entry
            dev_path="/dev/video2",
            cv2_index=1,
            model="Integrated Camera",
            vendor=None,
            serial=None,
        ),
    ]


def test_format_human_contains_paste_ready_snippet():
    text = _format_human(_sample_cameras(), probes=None)
    assert "Found 2 cameras:" in text
    assert "--device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_AAAA-video-index0" in text
    # The built-in camera falls back to the dev_path.
    assert "--device /dev/video2" in text
    # Integer index is present but subordinate.
    assert "cv2 index 0" in text
    assert "cv2 index 1" in text
    # The no-by-id case is annotated explicitly.
    assert "no by-id entry" in text


def test_format_human_no_cameras_message():
    text = _format_human([], probes=None)
    assert "No cameras found" in text


def test_format_human_with_probe():
    text = _format_human(
        _sample_cameras(),
        probes=[
            ProbeResult(ok=True, width=1920, height=1080, reason=None),
            ProbeResult(ok=False, width=None, height=None, reason="could not open"),
        ],
    )
    assert "1920x1080" in text
    assert "could not open" in text


def test_format_paths_one_per_line():
    text = _format_paths(_sample_cameras())
    lines = text.split("\n")
    assert lines == [
        "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_AAAA-video-index0",
        "/dev/video2",
    ]


def test_format_json_round_trip():
    import json
    text = _format_json(_sample_cameras(), probes=None)
    data = json.loads(text)
    assert len(data) == 2
    assert data[0]["model"] == "HD Pro Webcam C920"
    assert data[0]["serial"] == "AAAA"
    assert data[1]["by_id_path"] is None
    assert data[1]["dev_path"] == "/dev/video2"


# ----- dedup -----


def _info(dev: str, by_id: str | None = None) -> CameraInfo:
    return CameraInfo(
        by_id_path=by_id, dev_path=dev, cv2_index=-1,
        model="Stub", vendor=None, serial=None,
    )


def test_dedup_collapses_multi_interface_camera():
    # One physical USB device exposing two V4L2 capture nodes;
    # only the one with a by-id entry survives.
    rs_video4 = _info(
        "/dev/video4",
        by_id="/dev/v4l/by-id/usb-Intel_R__RealSense_TM_-video-index0",
    )
    rs_video8 = _info("/dev/video8", by_id=None)
    out = _dedup_by_usb([
        ("/sys/devices/.../5-2.4", rs_video4),
        ("/sys/devices/.../5-2.4", rs_video8),
    ])
    assert len(out) == 1
    assert out[0].dev_path == "/dev/video4"
    assert out[0].by_id_path is not None


def test_dedup_keeps_distinct_cameras_separate():
    cam_a = _info("/dev/video0", by_id="/dev/v4l/by-id/usb-A-video-index0")
    cam_b = _info("/dev/video1", by_id="/dev/v4l/by-id/usb-B-video-index0")
    out = _dedup_by_usb([
        ("/sys/devices/.../usb_a", cam_a),
        ("/sys/devices/.../usb_b", cam_b),
    ])
    assert len(out) == 2
    assert {c.dev_path for c in out} == {"/dev/video0", "/dev/video1"}


def test_dedup_prefers_by_id_then_lower_index():
    # Both nodes lack a by-id; the lower-indexed one wins.
    cam_video6 = _info("/dev/video6", by_id=None)
    cam_video10 = _info("/dev/video10", by_id=None)
    out = _dedup_by_usb([
        ("/sys/devices/.../5-3", cam_video10),
        ("/sys/devices/.../5-3", cam_video6),
    ])
    assert len(out) == 1
    assert out[0].dev_path == "/dev/video6"


def test_dedup_preserves_group_order_by_first_seen_index():
    later = _info("/dev/video4", by_id="/dev/v4l/by-id/usb-Late-video-index0")
    earlier = _info("/dev/video0", by_id="/dev/v4l/by-id/usb-Early-video-index0")
    out = _dedup_by_usb([
        ("/sys/devices/.../usb_late", later),
        ("/sys/devices/.../usb_early", earlier),
    ])
    # The group containing video0 (lower index) comes out first.
    assert [c.dev_path for c in out] == ["/dev/video0", "/dev/video4"]
