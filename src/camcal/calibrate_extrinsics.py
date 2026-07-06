"""Per-camera extrinsic against a planar calibration target.

The target defines the world frame ``W``. For each observation we solve
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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .aruco import ArucoMarkerConfig, ArucoMarkerDetection, ArucoMarkerTarget
from .camera import parse_device
from .charuco import CharucoBoard, CharucoConfig, CharucoDetection
from .io_yaml import (
    ExtrinsicsRecord,
    load_board,
    load_target_descriptor,
    profile_image_size,
    resolve_intrinsics_file,
    save_extrinsics,
)
from .list_cameras import CameraInfo, camera_block_from_device, resolve_device
from .pnp import reprojection_error_px, solve_pnp

DETECTOR_CHOICES = ("charuco", "aruco")
BoardObservation = CharucoDetection | ArucoMarkerDetection
DEFAULT_LIVE_WIDTH = 1280
DEFAULT_LIVE_HEIGHT = 720
DEFAULT_LIVE_FPS = 30


@dataclass
class CalibrationTarget:
    label: str
    detection_mode: str
    detector: object
    target: dict[str, object]
    board: CharucoConfig | None = None
    capture_board: CharucoBoard | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-extrinsics",
        description="Pose of a camera in a world frame defined by a planar target.",
    )
    p.add_argument(
        "--board",
        required=False,
        type=Path,
        help=(
            "Path to ChArUco board.yaml. Legacy shorthand for "
            "--target-descriptor with type: charuco_board."
        ),
    )
    p.add_argument(
        "--target-descriptor",
        type=Path,
        default=None,
        help=(
            "Target YAML. Supports type: charuco_board or type: aruco_marker "
            "with dictionary, marker_id, and size_m."
        ),
    )
    p.add_argument(
        "--intrinsics", required=False, type=Path,
        help=(
            "Path to intrinsics.yaml for this camera, or a "
            "realsense_intrinsics.yaml factory bundle. In --live mode on "
            "RealSense devices, omitting this will auto-try "
            "./realsense_intrinsics.yaml."
        ),
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Path to write extrinsics.yaml.",
    )
    p.add_argument(
        "--notes", default="",
        help="Free-text note saved alongside the result.",
    )
    p.add_argument(
        "--detector",
        choices=DETECTOR_CHOICES,
        default=None,
        help=(
            "Target detector. 'charuco' uses a ChArUco board. 'aruco' uses "
            "one standalone square marker, not markers embedded in a board."
        ),
    )
    p.add_argument(
        "--aruco-dictionary",
        default="DICT_5X5_100",
        help="Standalone marker dictionary, e.g. DICT_5X5_100.",
    )
    p.add_argument(
        "--aruco-marker-id",
        "--aruco-id",
        dest="aruco_marker_id",
        type=int,
        default=None,
        help="Standalone ArUco marker ID to detect.",
    )
    p.add_argument(
        "--aruco-size-m",
        type=float,
        default=None,
        help="Standalone ArUco marker side length in meters.",
    )
    p.add_argument(
        "--aruco-size-mm",
        type=float,
        default=None,
        help="Standalone ArUco marker side length in millimeters.",
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
    p.add_argument(
        "--device", type=parse_device, default=0,
        help="--live: OpenCV index (e.g. 0) or device path "
             "(e.g. /dev/v4l/by-id/usb-...-video-index0). Use by-id paths "
             "on multi-camera rigs — integer indices reshuffle on reboot.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="--live: capture width. Defaults to the factory profile when using RealSense factory intrinsics, otherwise 1280.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="--live: capture height. Defaults to the factory profile when using RealSense factory intrinsics, otherwise 720.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="--live: capture FPS. Defaults to the factory profile when using RealSense factory intrinsics, otherwise 30.",
    )
    p.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help=(
            "Select one camera from a RealSense factory intrinsics bundle. "
            "In --live mode this is usually inferred from --device."
        ),
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        intr, intrinsics_path, intrinsics_profile = _load_intrinsics_for_run(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: failed to resolve intrinsics: {exc}", file=sys.stderr)
        return 2

    try:
        target = _resolve_target(args)
    except Exception as exc:
        print(f"ERROR: failed to resolve target: {exc}", file=sys.stderr)
        return 2

    if args.live:
        try:
            width, height, fps = _resolve_live_capture_settings(
                args=args,
                intrinsics_profile=intrinsics_profile,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        try:
            detections = _detect_live(
                board=target.capture_board,
                detector=target.detector,
                detector_name=target.detection_mode,
                num_frames=args.num_frames,
                expected_image_size=intr.image_size,
                device=args.device, width=width, height=height, fps=fps,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if detections is None:
            print("camcal-extrinsics: aborted (ESC).", file=sys.stderr)
            return 1
    elif args.image is not None:
        detections = _detect_from_paths(
            paths=[str(args.image)],
            detector=target.detector,
            expected_image_size=intr.image_size,
        )
    else:
        paths = sorted(glob.glob(args.images))
        if not paths:
            print(
                f"ERROR: --images glob matched no files: {args.images!r}",
                file=sys.stderr,
            )
            return 2
        detections = _detect_from_paths(
            paths=paths,
            detector=target.detector,
            expected_image_size=intr.image_size,
        )

    if not detections:
        print(
            "ERROR: no successful target detections; cannot solve extrinsics.",
            file=sys.stderr,
        )
        return 2

    per_frame_W_T_C: list[np.ndarray] = []
    per_frame_reproj_mean: list[float] = []
    per_frame_reproj_max: list[float] = []
    for det in detections:
        pnp_method = (
            cv2.SOLVEPNP_IPPE_SQUARE
            if isinstance(det, ArucoMarkerDetection)
            else cv2.SOLVEPNP_ITERATIVE
        )
        C_T_W = solve_pnp(
            det.object_points,
            det.image_points,
            intr.K,
            intr.d,
            method=pnp_method,
        )
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
        n_frames=len(per_frame_W_T_C),
        reprojection_error_px_mean=float(np.mean(per_frame_reproj_mean)),
        reprojection_error_px_max=float(np.max(per_frame_reproj_max)),
        board=target.board,
        target=target.target,
        intrinsics_path=str(intrinsics_path),
        detection_mode=target.detection_mode,
        notes=args.notes,
        camera=_camera_block(args),
        per_frame_W_T_C=per_frame_W_T_C if len(per_frame_W_T_C) > 1 else [],
    )
    save_extrinsics(args.out, record)
    t = W_T_C_mean[:3, 3]
    print(
        f"camcal-extrinsics: wrote {args.out} — W_T_C translation "
        f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m, "
        f"{target.label}, mean reprojection "
        f"{record.reprojection_error_px_mean:.3f} px over "
        f"{record.n_frames} frame(s).",
        file=sys.stderr,
    )
    return 0


# ----- provenance -----


def _camera_block(args: argparse.Namespace) -> dict[str, object]:
    """Build the ``camera:`` block recorded into extrinsics.yaml."""
    if args.live:
        return camera_block_from_device(args.device)
    if args.image is not None:
        return {"source": "single_image", "path": str(args.image)}
    return {"source": "files", "glob": args.images}


def _resolve_target(args: argparse.Namespace) -> CalibrationTarget:
    descriptor = _load_target_from_args(args)
    target_type = str(descriptor["type"])
    if target_type == "charuco_board":
        board = CharucoConfig(
            squares_x=int(descriptor["squares_x"]),
            squares_y=int(descriptor["squares_y"]),
            square_size_m=float(descriptor["square_size_m"]),
            marker_size_m=float(descriptor["marker_size_m"]),
            dictionary=str(descriptor.get("dictionary", "DICT_5X5_100")),
        )
        runtime = CharucoBoard.from_config(board)
        return CalibrationTarget(
            label="charuco target",
            detection_mode="charuco",
            detector=runtime.detect_full,
            target=descriptor,
            board=board,
            capture_board=runtime,
        )
    if target_type == "aruco_marker":
        config = ArucoMarkerConfig(
            dictionary=str(descriptor.get("dictionary", "DICT_5X5_100")),
            marker_id=int(descriptor["marker_id"]),
            size_m=float(descriptor["size_m"]),
        )
        runtime = ArucoMarkerTarget(config)
        return CalibrationTarget(
            label=(
                f"aruco marker id={config.marker_id} "
                f"{config.dictionary} size={config.size_m:.6f}m"
            ),
            detection_mode="aruco_marker",
            detector=runtime.detect,
            target=config.to_target_dict(),
        )
    raise ValueError(f"unsupported target type {target_type!r}")


def _load_target_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.target_descriptor is not None:
        return load_target_descriptor(args.target_descriptor)

    wants_aruco = (
        args.detector == "aruco"
        or args.aruco_marker_id is not None
        or args.aruco_size_m is not None
        or args.aruco_size_mm is not None
    )
    if wants_aruco:
        size_m = _resolve_aruco_size_m(args)
        if args.aruco_marker_id is None:
            raise ValueError("--aruco-marker-id is required for standalone ArUco")
        if size_m is None:
            raise ValueError("--aruco-size-m or --aruco-size-mm is required for standalone ArUco")
        return {
            "type": "aruco_marker",
            "dictionary": str(args.aruco_dictionary),
            "marker_id": int(args.aruco_marker_id),
            "size_m": size_m,
            "origin": "center",
        }

    if args.detector not in (None, "charuco"):
        raise ValueError(f"unsupported detector {args.detector!r}")
    if args.board is None:
        raise ValueError("--board or --target-descriptor is required for ChArUco")
    board = load_board(args.board)
    return {
        "type": "charuco_board",
        "squares_x": board.squares_x,
        "squares_y": board.squares_y,
        "square_size_m": board.square_size_m,
        "marker_size_m": board.marker_size_m,
        "dictionary": board.dictionary,
    }


def _resolve_aruco_size_m(args: argparse.Namespace) -> float | None:
    if args.aruco_size_m is not None and args.aruco_size_mm is not None:
        raise ValueError("provide only one of --aruco-size-m or --aruco-size-mm")
    if args.aruco_size_m is not None:
        return float(args.aruco_size_m)
    if args.aruco_size_mm is not None:
        return float(args.aruco_size_mm) / 1000.0
    return None


def _load_intrinsics_for_run(
    args: argparse.Namespace,
):
    explicit_path = args.intrinsics
    selector_serial = _intrinsics_selector_serial(args)
    if explicit_path is not None:
        intr, profile = resolve_intrinsics_file(
            explicit_path,
            camera_serial=selector_serial,
        )
        return intr, explicit_path, profile

    auto_path = _default_realsense_intrinsics_path(args)
    if auto_path is not None:
        intr, profile = resolve_intrinsics_file(
            auto_path,
            camera_serial=selector_serial,
        )
        print(
            f"INFO: using RealSense factory intrinsics from {auto_path}",
            file=sys.stderr,
        )
        return intr, auto_path, profile

    raise FileNotFoundError(
        "--intrinsics is required unless --live is targeting a RealSense "
        "camera with ./realsense_intrinsics.yaml available."
    )


def _intrinsics_selector_serial(args: argparse.Namespace) -> str | None:
    if args.camera_serial:
        return str(args.camera_serial)
    info = _resolved_live_camera(args)
    if info is not None and info.serial:
        return info.serial
    return None


def _resolved_live_camera(args: argparse.Namespace) -> CameraInfo | None:
    if not args.live:
        return None
    return resolve_device(args.device)


def _default_realsense_intrinsics_path(args: argparse.Namespace) -> Path | None:
    info = _resolved_live_camera(args)
    if info is None or not _looks_like_realsense(info):
        return None
    path = Path("realsense_intrinsics.yaml")
    if path.is_file():
        return path
    return None


def _looks_like_realsense(info: CameraInfo) -> bool:
    model = (info.model or "").lower()
    vendor = (info.vendor or "").lower()
    return "realsense" in model or (vendor == "intel" and info.serial is not None)


def _resolve_live_capture_settings(
    *,
    args: argparse.Namespace,
    intrinsics_profile: dict[str, object] | None,
) -> tuple[int, int, int]:
    if intrinsics_profile is None:
        return (
            int(args.width if args.width is not None else DEFAULT_LIVE_WIDTH),
            int(args.height if args.height is not None else DEFAULT_LIVE_HEIGHT),
            int(args.fps if args.fps is not None else DEFAULT_LIVE_FPS),
        )

    profile_width, profile_height = profile_image_size(intrinsics_profile)
    profile_fps = int(intrinsics_profile.get("fps", DEFAULT_LIVE_FPS))
    width = int(args.width if args.width is not None else profile_width)
    height = int(args.height if args.height is not None else profile_height)
    fps = int(args.fps if args.fps is not None else profile_fps)
    if (width, height, fps) != (profile_width, profile_height, profile_fps):
        raise ValueError(
            "RealSense factory intrinsics are profile-specific. Use "
            f"{profile_width}x{profile_height}@{profile_fps} to match the "
            "selected factory profile."
        )
    return width, height, fps


# ----- input modes -----


def _detect_from_paths(
    *,
    paths: list[str],
    detector,
    expected_image_size: tuple[int, int],
) -> list[BoardObservation]:
    out: list[BoardObservation] = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"WARN: cannot read {path}", file=sys.stderr)
            continue
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if size != expected_image_size:
            print(
                f"WARN: {path} size {size} does not match intrinsics image size "
                f"{expected_image_size}; skipping.",
                file=sys.stderr,
            )
            continue
        det = detector(frame)
        if det is None:
            print(f"WARN: no board detected in {path}", file=sys.stderr)
            continue
        out.append(det)
    return out


def _detect_live(
    *,
    board: CharucoBoard | None,
    detector,
    detector_name: str,
    num_frames: int,
    expected_image_size: tuple[int, int],
    device: int | str, width: int, height: int, fps: int,
) -> list[BoardObservation] | None:
    """Open the webcam and run the SPACE/ENTER/ESC loop."""
    from ._capture_loop import run_capture_loop
    from .camera import Camera

    detections: list[BoardObservation] = []

    def _on_capture(
        _frame: np.ndarray,
        det: BoardObservation,
    ) -> str | None:
        detections.append(det)
        return None

    with Camera(device=device, width=width, height=height, fps=fps) as cam:
        actual_size = cam.image_size()
        if actual_size != expected_image_size:
            raise ValueError(
                f"camera is streaming at {actual_size}, but intrinsics expect "
                f"{expected_image_size}"
            )
        result = run_capture_loop(
            label=f"extrinsics ({detector_name})",
            camera=cam,
            board=board,
            detector=detector,
            min_captures=num_frames,
            capture_count=lambda: len(detections),
            on_capture=_on_capture,
        )
    if result.exit_code != 0:
        return None
    return detections


if __name__ == "__main__":
    sys.exit(main())
