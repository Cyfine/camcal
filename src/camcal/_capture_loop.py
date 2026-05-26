"""Shared SPACE/ENTER/ESC live capture loop used by both scripts.

The loop opens a single OpenCV window, polls the camera, and overlays
detection feedback. Three keys:

* ``SPACE`` — invoke ``on_capture(frame, detection)``. If it returns a
  reason string, the capture is rejected and shown in the header band.
  If it returns ``None``, the capture is accepted.
* ``ENTER`` — finalise (succeeds once ``capture_count()`` ≥ ``min_captures``).
* ``ESC`` — abort.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .charuco import CharucoBoard, CharucoDetection

KEY_ESC = 27
KEY_ENTER = 13
KEY_SPACE = 32

_WINDOW_TITLE_PREFIX = "camcal"


@dataclass
class LoopResult:
    """Outcome of :func:`run_capture_loop`."""

    exit_code: int                       # 0 success, 1 ESC abort
    image_size: tuple[int, int] | None   # (width, height) of the last frame


def run_capture_loop(
    *,
    label: str,
    camera: object,
    board: CharucoBoard,
    min_captures: int,
    capture_count: Callable[[], int],
    on_capture: Callable[[np.ndarray, CharucoDetection], str | None],
) -> LoopResult:
    """Drive the SPACE/ENTER/ESC live loop.

    Returns once the operator presses ENTER (with enough captures) or
    ESC. ``camera`` must already be connected; the caller owns its
    lifecycle.
    """
    title = f"{_WINDOW_TITLE_PREFIX}: {label}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last_msg = ""
    warning = ""
    image_size: tuple[int, int] | None = None
    try:
        while True:
            frame = camera.capture()  # type: ignore[attr-defined]
            if frame is None or frame.size == 0:
                continue
            image_size = (int(frame.shape[1]), int(frame.shape[0]))
            detection = board.detect_full(frame)
            n_corners = (
                int(detection.object_points.shape[0]) if detection else 0
            )

            display = frame.copy()
            if detection is not None:
                _draw_corner_dots(display, detection.image_points)

            header = [
                f"{label}: {capture_count()}/{min_captures} captured  "
                "(SPACE=capture, ENTER=finalise, ESC=abort)",
                (
                    f"detected {n_corners} corners — ready"
                    if detection is not None
                    else "no board detected — adjust framing"
                ),
            ]
            if last_msg:
                header.append(f"last: {last_msg}")
            if warning:
                header.append(f"! {warning}")
            _draw_header(display, header)

            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF

            if key == KEY_ESC:
                return LoopResult(exit_code=1, image_size=image_size)

            if key == KEY_SPACE:
                if detection is None:
                    warning = "no board detected; adjust + retry"
                    continue
                warning = ""
                reason = on_capture(frame.copy(), detection)
                if reason is not None:
                    warning = reason
                    continue
                last_msg = f"captured {capture_count()}, corners={n_corners}"
                continue

            if key == KEY_ENTER:
                if capture_count() < min_captures:
                    warning = (
                        f"need {min_captures - capture_count()} more captures"
                    )
                    continue
                return LoopResult(exit_code=0, image_size=image_size)
    finally:
        cv2.destroyWindow(title)


# ----- overlay helpers -----


def _draw_corner_dots(img: np.ndarray, image_points: np.ndarray) -> None:
    for pt in image_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)


def _draw_header(img: np.ndarray, lines: list[str]) -> None:
    pad_x, pad_y, line_h = 12, 24, 22
    box_h = pad_y + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (pad_x, pad_y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


__all__ = ["KEY_ENTER", "KEY_ESC", "KEY_SPACE", "LoopResult", "run_capture_loop"]
