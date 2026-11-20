"""USB/UVC webcam adapter built on ``cv2.VideoCapture``.

Use as a context manager::

    with Camera(device=0, width=1920, height=1080) as cam:
        frame = cam.capture()
"""

from __future__ import annotations

import time

import cv2
import numpy as np

# How long to wait for a single frame before retrying once.
_FRAME_TIMEOUT_S = 2.0
_REALSENSE_FRAME_TIMEOUT_MS = 5000


def _read(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Read one frame; an undecodable frame counts as a failed read.

    OpenCV 5 raises from ``read()`` when a UVC camera delivers an empty MJPG
    buffer, where OpenCV 4 returned ``ok=False``.
    """
    try:
        return cap.read()
    except cv2.error:
        return False, None


# Brief warm-up: many UVC cameras return black or stale frames for the
# first few reads while exposure converges.
_WARMUP_FRAMES = 5


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a capture fails."""


def parse_device(value: str) -> int | str:
    """Argparse type for ``--device``: integer index or device path.

    Integer indices (``"0"``, ``"1"``) become OpenCV ``VideoCapture``
    indices. On Linux these are unstable across reboots — the kernel
    enumerates ``/dev/videoN`` in USB discovery order.

    Pass a stable path instead when calibrating a rig of two or more
    cameras, e.g.::

        --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0

    These ``by-id`` paths are keyed off the USB device serial and stay
    fixed across reboots and ports.
    """
    s = value.strip()
    if s.isdigit():
        return int(s)
    return s


class Camera:
    """Open a webcam, configure resolution, and capture single frames.

    Parameters
    ----------
    device
        OpenCV device index (typically ``0`` for the first webcam) or a
        gstreamer/video URI string.
    width, height
        Requested capture resolution. The driver may snap to the nearest
        supported mode; the actual size is exposed via :meth:`image_size`
        once the device is open.
    fps
        Requested frame rate. Best-effort; many cameras ignore this.
    fourcc
        Optional four-character codec, e.g. ``"MJPG"``. Selecting MJPG on
        UVC cameras unlocks higher resolutions at higher frame rates than
        the default YUYV mode.
    """

    def __init__(
        self,
        *,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fourcc: str | None = "MJPG",
    ) -> None:
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._fourcc = fourcc
        self._cap: cv2.VideoCapture | None = None

    # ----- lifecycle -----

    def connect(self) -> None:
        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise CameraError(f"could not open video device {self._device!r}")
        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Keep only the newest frame queued, so a read returns the present, not a stale buffer.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        for _ in range(_WARMUP_FRAMES):
            _read(cap)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.disconnect()

    # ----- capture -----

    def capture(self) -> np.ndarray:
        """Read one BGR frame as a ``(H, W, 3)`` uint8 array."""
        if self._cap is None:
            raise CameraError("camera is not connected; call connect() first")
        deadline = time.monotonic() + _FRAME_TIMEOUT_S
        last_ok = False
        last_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            ok, frame = _read(self._cap)
            if ok and frame is not None and frame.size > 0:
                last_ok, last_frame = True, frame
                break
        if not last_ok or last_frame is None:
            self.disconnect()
            self.connect()
            ok, frame = _read(self._cap) if self._cap is not None else (False, None)
            if not ok or frame is None:
                raise CameraError(
                    f"capture failed on device {self._device!r} "
                    f"(within {_FRAME_TIMEOUT_S:.1f}s + one reconnect)"
                )
            last_frame = frame
        return last_frame

    def grab(self) -> bool:
        """Take the current frame without decoding it; pair with :meth:`retrieve`.

        Grabbing every camera first and decoding afterwards keeps the frames of a
        multi-camera capture as close in time as unsynchronised cameras allow.
        """
        if self._cap is None:
            raise CameraError("camera is not connected; call connect() first")
        try:
            return bool(self._cap.grab())
        except cv2.error:
            return False

    def retrieve(self) -> np.ndarray:
        """Decode the frame taken by :meth:`grab`, or read a new one if that failed."""
        if self._cap is None:
            raise CameraError("camera is not connected; call connect() first")
        try:
            ok, frame = self._cap.retrieve()
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and frame.size > 0:
            return frame
        return self.capture()

    def image_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` actually reported by the driver."""
        if self._cap is None:
            raise CameraError("camera is not connected; call connect() first")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h


class RealSenseCamera:
    """Capture one Intel RealSense color stream through librealsense."""

    def __init__(
        self,
        *,
        serial: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._serial = str(serial)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pipeline = None

    def connect(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "RealSense capture requires pyrealsense2; install the camcal "
                "realsense extra"
            ) from exc
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self._serial)
        config.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.bgr8,
            self._fps,
        )
        try:
            pipeline.start(config)
            for _ in range(_WARMUP_FRAMES):
                pipeline.wait_for_frames(timeout_ms=_REALSENSE_FRAME_TIMEOUT_MS)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise CameraError(
                f"could not start RealSense {self._serial} color stream "
                f"at {self._width}x{self._height}@{self._fps}: {exc}"
            ) from exc
        self._pipeline = pipeline

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def capture(self) -> np.ndarray:
        if self._pipeline is None:
            raise CameraError("camera is not connected; call connect() first")
        try:
            frames = self._pipeline.wait_for_frames(
                timeout_ms=_REALSENSE_FRAME_TIMEOUT_MS
            )
        except Exception as exc:
            raise CameraError(
                f"capture failed on RealSense {self._serial}: {exc}"
            ) from exc
        color = frames.get_color_frame()
        if not color:
            raise CameraError(f"RealSense {self._serial} returned no color frame")
        frame = np.asanyarray(color.get_data())
        if frame.shape != (self._height, self._width, 3):
            raise CameraError(
                f"RealSense {self._serial} returned frame shape {frame.shape}, "
                f"expected {(self._height, self._width, 3)}"
            )
        return frame.copy()

    def grab(self) -> bool:
        """Take the next colour frame now; :meth:`retrieve` returns it."""
        self._grabbed = self.capture()
        return True

    def retrieve(self) -> np.ndarray:
        frame, self._grabbed = getattr(self, "_grabbed", None), None
        return frame if frame is not None else self.capture()

    def image_size(self) -> tuple[int, int]:
        if self._pipeline is None:
            raise CameraError("camera is not connected; call connect() first")
        return self._width, self._height


__all__ = ["Camera", "CameraError", "RealSenseCamera", "parse_device"]
