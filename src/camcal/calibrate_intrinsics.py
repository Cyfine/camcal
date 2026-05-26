"""Zhang's-method camera intrinsic calibration.

Two input modes:

* ``--images "captures/*.png"`` — run over pre-captured image files.
* ``--live --device 0`` — open a webcam, SPACE/ENTER/ESC loop.

Writes an ``intrinsics.yaml`` (see :mod:`camcal.io_yaml`).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoard
from .intrinsics import MIN_VIEWS, calibrate_intrinsics
from .io_yaml import IntrinsicsRecord, load_board, save_intrinsics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-intrinsics",
        description="Zhang's-method camera intrinsics from ChArUco views.",
    )
    p.add_argument(
        "--board", required=True, type=Path,
        help="Path to board.yaml (caliper-measured ChArUco geometry).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Path to write intrinsics.yaml.",
    )
    p.add_argument(
        "--notes", default="",
        help="Free-text note saved alongside the result.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--images", type=str, default=None,
        help="Glob pattern of input image files (PNG/JPG/...).",
    )
    src.add_argument(
        "--live", action="store_true",
        help="Open a webcam and capture views interactively.",
    )

    p.add_argument(
        "--min-views", type=int, default=15,
        help="Minimum captures before ENTER finalises in --live mode "
             "(default 15). Zhang's wants 10+ from varied angles.",
    )

    # Live-only args.
    p.add_argument("--device", type=int, default=0, help="--live: device index.")
    p.add_argument("--width", type=int, default=1280, help="--live: capture width.")
    p.add_argument("--height", type=int, default=720, help="--live: capture height.")
    p.add_argument("--fps", type=int, default=30, help="--live: capture FPS.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_board(args.board)
    except FileNotFoundError:
        print(f"ERROR: board.yaml not found: {args.board}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: failed to load {args.board}: {exc}", file=sys.stderr)
        return 2

    board = CharucoBoard.from_config(config)

    if args.live:
        image_points, object_points, image_size = _capture_live(
            board=board, min_views=args.min_views,
            device=args.device, width=args.width, height=args.height, fps=args.fps,
        )
    else:
        paths = sorted(glob.glob(args.images))
        if not paths:
            print(
                f"ERROR: --images glob matched no files: {args.images!r}",
                file=sys.stderr,
            )
            return 2
        image_points, object_points, image_size = _capture_from_images(board, paths)

    if image_points is None:
        return 1  # ESC abort from live mode

    if len(image_points) < MIN_VIEWS:
        print(
            f"ERROR: need at least {MIN_VIEWS} valid views with detections, "
            f"got {len(image_points)}.",
            file=sys.stderr,
        )
        return 2

    try:
        result = calibrate_intrinsics(
            image_points=image_points,
            object_points=object_points,
            image_size=image_size,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    record = IntrinsicsRecord(
        K=result.K,
        d=result.d,
        image_size=result.image_size,
        reprojection_error_px=result.reprojection_error_px,
        n_views=len(image_points),
        source="zhang",
        notes=args.notes,
    )
    save_intrinsics(args.out, record)
    print(
        f"camcal-intrinsics: wrote {args.out} "
        f"(RMS reprojection {result.reprojection_error_px:.3f} px over "
        f"{len(image_points)} views)",
        file=sys.stderr,
    )
    return 0


# ----- input modes -----


def _capture_from_images(
    board: CharucoBoard, paths: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect the board in each file. Skip files with no detection (with a warning)."""
    image_points: list[np.ndarray] = []
    object_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"WARN: cannot read {path}", file=sys.stderr)
            continue
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(
                f"WARN: {path} size {size} differs from first image {image_size}; "
                "skipping (all views must share a resolution).",
                file=sys.stderr,
            )
            continue
        det = board.detect_full(frame)
        if det is None:
            print(f"WARN: no board detected in {path}", file=sys.stderr)
            continue
        image_points.append(det.image_points)
        object_points.append(det.object_points)
    if image_size is None:
        # All reads failed.
        image_size = (0, 0)
    return image_points, object_points, image_size


def _capture_live(
    *,
    board: CharucoBoard,
    min_views: int,
    device: int, width: int, height: int, fps: int,
) -> tuple[list[np.ndarray] | None, list[np.ndarray], tuple[int, int]]:
    """Open the webcam and run the SPACE/ENTER/ESC loop."""
    from ._capture_loop import run_capture_loop
    from .camera import Camera

    image_points: list[np.ndarray] = []
    object_points: list[np.ndarray] = []

    def _on_capture(_frame: np.ndarray, det) -> str | None:  # noqa: ANN001
        image_points.append(det.image_points)
        object_points.append(det.object_points)
        return None

    with Camera(device=device, width=width, height=height, fps=fps) as cam:
        result = run_capture_loop(
            label="intrinsics",
            camera=cam,
            board=board,
            min_captures=min_views,
            capture_count=lambda: len(image_points),
            on_capture=_on_capture,
        )
        size = cam.image_size()
    if result.exit_code != 0:
        print("camcal-intrinsics: aborted (ESC).", file=sys.stderr)
        return None, [], size
    return image_points, object_points, size


if __name__ == "__main__":
    sys.exit(main())
