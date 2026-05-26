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
# Brief warm-up: many UVC cameras return black or stale frames for the
# first few reads while exposure converges.
_WARMUP_FRAMES = 5


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a capture fails."""


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
        self._cap = cap
        for _ in range(_WARMUP_FRAMES):
            cap.read()

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
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                last_ok, last_frame = True, frame
                break
        if not last_ok or last_frame is None:
            self.disconnect()
            self.connect()
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            if not ok or frame is None:
                raise CameraError(
                    f"capture failed on device {self._device!r} "
                    f"(within {_FRAME_TIMEOUT_S:.1f}s + one reconnect)"
                )
            last_frame = frame
        return last_frame

    def image_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` actually reported by the driver."""
        if self._cap is None:
            raise CameraError("camera is not connected; call connect() first")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h


__all__ = ["Camera", "CameraError"]
