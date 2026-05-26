from camcal.camera import parse_device


def test_parse_device_integer_index():
    assert parse_device("0") == 0
    assert parse_device("1") == 1
    assert parse_device("12") == 12


def test_parse_device_passes_through_path():
    path = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_ABCD-video-index0"
    assert parse_device(path) == path


def test_parse_device_passes_through_dev_video_path():
    assert parse_device("/dev/video2") == "/dev/video2"


def test_parse_device_strips_whitespace():
    assert parse_device("  0  ") == 0
    assert parse_device("  /dev/video0  ") == "/dev/video0"
