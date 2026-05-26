"""Per-camera extrinsic against a ChArUco-defined world frame.

The board defines the world frame ``W``. For each observation we solve
PnP to get ``C_T_W``, invert to ``W_T_C``, then geodesic-mean across all
observations. Writes an ``extrinsics.yaml`` (see :mod:`camcal.io_yaml`).

Three input modes:

* ``--image path.png`` — a single image (no averaging).
* ``--images "views/*.png"`` — multiple pre-captured images, averaged.
* ``--live --device 0 --num-frames N`` — webcam, SPACE/ENTER/ESC, averaged.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .charuco import CharucoBoard, CharucoDetection
from .io_yaml import (
    ExtrinsicsRecord,
    load_board,
    load_intrinsics,
    save_extrinsics,
)
from .pnp import reprojection_error_px, solve_pnp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-extrinsics",
        description="Pose of a camera in a world frame defined by a ChArUco board.",
    )
    p.add_argument(
        "--board", required=True, type=Path,
        help="Path to board.yaml (caliper-measured ChArUco geometry).",
    )
    p.add_argument(
        "--intrinsics", required=True, type=Path,
        help="Path to intrinsics.yaml for this camera.",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Path to write extrinsics.yaml.",
    )
    p.add_argument(
        "--notes", default="",
        help="Free-text note saved alongside the result.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=Path, default=None,
        help="A single image path. Convenience for the one-shot case.",
    )
    src.add_argument(
        "--images", type=str, default=None,
        help="Glob of multiple images; the W_T_C is geodesic-averaged.",
    )
    src.add_argument(
        "--live", action="store_true",
        help="Open a webcam and capture views interactively.",
    )

    p.add_argument(
        "--num-frames", type=int, default=5,
        help="--live only: minimum captures before ENTER finalises "
             "(default 5).",
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
        intr = load_intrinsics(args.intrinsics)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: failed to load inputs: {exc}", file=sys.stderr)
        return 2

    board = CharucoBoard.from_config(config)

    if args.live:
        detections = _detect_live(
            board=board,
            num_frames=args.num_frames,
            device=args.device, width=args.width, height=args.height, fps=args.fps,
        )
        if detections is None:
            print("camcal-extrinsics: aborted (ESC).", file=sys.stderr)
            return 1
    elif args.image is not None:
        detections = _detect_from_paths(board, [str(args.image)])
    else:
        paths = sorted(glob.glob(args.images))
        if not paths:
            print(
                f"ERROR: --images glob matched no files: {args.images!r}",
                file=sys.stderr,
            )
            return 2
        detections = _detect_from_paths(board, paths)

    if not detections:
        print(
            "ERROR: no successful board detections; cannot solve extrinsics.",
            file=sys.stderr,
        )
        return 2

    per_frame_W_T_C: list[np.ndarray] = []
    per_frame_reproj_mean: list[float] = []
    per_frame_reproj_max: list[float] = []
    for det in detections:
        C_T_W = solve_pnp(det.object_points, det.image_points, intr.K, intr.d)
        if C_T_W is None:
            print("WARN: PnP failed on one view; skipping.", file=sys.stderr)
            continue
        W_T_C = se3.inverse(C_T_W)
        per_frame_W_T_C.append(W_T_C)
        mean_px, max_px = reprojection_error_px(
            object_points=det.object_points,
            image_points=det.image_points,
            K=intr.K, d=intr.d, C_T_T=C_T_W,
        )
        per_frame_reproj_mean.append(mean_px)
        per_frame_reproj_max.append(max_px)

    if not per_frame_W_T_C:
        print("ERROR: every PnP solve failed.", file=sys.stderr)
        return 2

    W_T_C_mean = (
        per_frame_W_T_C[0] if len(per_frame_W_T_C) == 1
        else se3.mean_se3(per_frame_W_T_C)
    )

    record = ExtrinsicsRecord(
        W_T_C=W_T_C_mean,
        board=config,
        n_frames=len(per_frame_W_T_C),
        reprojection_error_px_mean=float(np.mean(per_frame_reproj_mean)),
        reprojection_error_px_max=float(np.max(per_frame_reproj_max)),
        intrinsics_path=str(args.intrinsics),
        notes=args.notes,
        per_frame_W_T_C=per_frame_W_T_C if len(per_frame_W_T_C) > 1 else [],
    )
    save_extrinsics(args.out, record)
    t = W_T_C_mean[:3, 3]
    print(
        f"camcal-extrinsics: wrote {args.out} — W_T_C translation "
        f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m, "
        f"mean reprojection {record.reprojection_error_px_mean:.3f} px over "
        f"{record.n_frames} frame(s).",
        file=sys.stderr,
    )
    return 0


# ----- input modes -----


def _detect_from_paths(
    board: CharucoBoard, paths: list[str],
) -> list[CharucoDetection]:
    out: list[CharucoDetection] = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"WARN: cannot read {path}", file=sys.stderr)
            continue
        det = board.detect_full(frame)
        if det is None:
            print(f"WARN: no board detected in {path}", file=sys.stderr)
            continue
        out.append(det)
    return out


def _detect_live(
    *,
    board: CharucoBoard,
    num_frames: int,
    device: int, width: int, height: int, fps: int,
) -> list[CharucoDetection] | None:
    """Open the webcam and run the SPACE/ENTER/ESC loop."""
    from ._capture_loop import run_capture_loop
    from .camera import Camera

    detections: list[CharucoDetection] = []

    def _on_capture(_frame: np.ndarray, det: CharucoDetection) -> str | None:
        detections.append(det)
        return None

    with Camera(device=device, width=width, height=height, fps=fps) as cam:
        result = run_capture_loop(
            label="extrinsics",
            camera=cam,
            board=board,
            min_captures=num_frames,
            capture_count=lambda: len(detections),
            on_capture=_on_capture,
        )
    if result.exit_code != 0:
        return None
    return detections


if __name__ == "__main__":
    sys.exit(main())
