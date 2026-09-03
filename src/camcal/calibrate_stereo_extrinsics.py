"""One-shot relative extrinsics from a target seen by two live cameras.

The two cameras are read concurrently.  A single accepted image pair yields
the pose of the same target in each camera, from which the camera-to-camera
transform is composed.  This removes the requirement that a handheld target
remain fixed while two cameras are calibrated sequentially.

UVC cameras are not hardware synchronised; the operator must hold the target
still for the instant in which SPACE is pressed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .aruco import ArucoMarkerConfig, ArucoMarkerDetection, ArucoMarkerTarget
from .camera import Camera, parse_device
from .io_yaml import ExtrinsicsRecord, load_intrinsics, save_extrinsics
from .list_cameras import camera_block_from_device
from .pnp import reprojection_error_px, solve_pnp


def relative_camera_transform(
    C0_T_target: np.ndarray,
    C1_T_target: np.ndarray,
) -> np.ndarray:
    """Return ``C1_T_C0`` from the target pose observed by both cameras."""
    se3.assert_se3(C0_T_target)
    se3.assert_se3(C1_T_target)
    return C1_T_target @ se3.inverse(C0_T_target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camcal-stereo-extrinsics",
        description=(
            "Capture one concurrent stereo pair and solve camera-to-camera "
            "extrinsics from a shared ArUco marker."
        ),
    )
    parser.add_argument("--device-0", required=True, type=parse_device)
    parser.add_argument("--device-1", required=True, type=parse_device)
    parser.add_argument("--intrinsics-0", required=True, type=Path)
    parser.add_argument("--intrinsics-1", required=True, type=Path)
    parser.add_argument("--out-0", required=True, type=Path)
    parser.add_argument("--out-1", required=True, type=Path)
    parser.add_argument("--aruco-dictionary", default="DICT_5X5_100")
    parser.add_argument("--aruco-marker-id", type=int, required=True)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--aruco-size-m", type=float)
    size.add_argument("--aruco-size-mm", type=float)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def _capture_pair(
    executor: ThreadPoolExecutor,
    camera_0: Camera,
    camera_1: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    future_0 = executor.submit(camera_0.capture)
    future_1 = executor.submit(camera_1.capture)
    return future_0.result(), future_1.result()


def _annotate(
    frame: np.ndarray,
    detection: ArucoMarkerDetection | None,
    label: str,
) -> np.ndarray:
    shown = frame.copy()
    color = (0, 220, 0) if detection is not None else (0, 0, 255)
    state = "detected" if detection is not None else "not detected"
    cv2.putText(
        shown,
        f"{label}: {state}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    if detection is not None:
        for point in detection.image_points:
            cv2.circle(shown, tuple(np.rint(point).astype(int)), 6, color, -1)
    return shown


def _preview(frame_0: np.ndarray, frame_1: np.ndarray) -> np.ndarray:
    target_height = 540
    tiles = []
    for frame in (frame_0, frame_1):
        scale = target_height / frame.shape[0]
        tiles.append(
            cv2.resize(
                frame,
                (int(round(frame.shape[1] * scale)), target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    canvas = cv2.hconcat(tiles)
    cv2.putText(
        canvas,
        "SPACE: accept this pair    ESC/q: abort",
        (24, canvas.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _solve_target_pose(
    detection: ArucoMarkerDetection,
    K: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    pose = solve_pnp(
        detection.object_points,
        detection.image_points,
        K,
        d,
        method=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if pose is None:
        raise RuntimeError("PnP failed")
    mean_px, max_px = reprojection_error_px(
        object_points=detection.object_points,
        image_points=detection.image_points,
        K=K,
        d=d,
        C_T_T=pose,
    )
    return pose, mean_px, max_px


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    size_m = (
        float(args.aruco_size_m)
        if args.aruco_size_m is not None
        else float(args.aruco_size_mm) / 1000.0
    )
    target_config = ArucoMarkerConfig(
        dictionary=args.aruco_dictionary,
        marker_id=args.aruco_marker_id,
        size_m=size_m,
    )
    target = ArucoMarkerTarget(target_config)
    intrinsics_0 = load_intrinsics(args.intrinsics_0)
    intrinsics_1 = load_intrinsics(args.intrinsics_1)

    accepted: tuple[ArucoMarkerDetection, ArucoMarkerDetection] | None = None
    try:
        with (
            Camera(
                device=args.device_0,
                width=args.width,
                height=args.height,
                fps=args.fps,
            ) as camera_0,
            Camera(
                device=args.device_1,
                width=args.width,
                height=args.height,
                fps=args.fps,
            ) as camera_1,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            expected = (args.width, args.height)
            for index, camera in enumerate((camera_0, camera_1)):
                if camera.image_size() != expected:
                    raise ValueError(
                        f"camera {index} streams at {camera.image_size()}, "
                        f"expected {expected}"
                    )
            if intrinsics_0.image_size != expected:
                raise ValueError(
                    f"camera 0 intrinsics expect {intrinsics_0.image_size}, "
                    f"stream is {expected}"
                )
            if intrinsics_1.image_size != expected:
                raise ValueError(
                    f"camera 1 intrinsics expect {intrinsics_1.image_size}, "
                    f"stream is {expected}"
                )

            while True:
                frame_0, frame_1 = _capture_pair(executor, camera_0, camera_1)
                detection_0 = target.detect(frame_0)
                detection_1 = target.detect(frame_1)
                canvas = _preview(
                    _annotate(frame_0, detection_0, "camera 0"),
                    _annotate(frame_1, detection_1, "camera 1"),
                )
                cv2.imshow("camcal: stereo extrinsics one-shot", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 1
                if key == ord(" "):
                    if detection_0 is None or detection_1 is None:
                        print(
                            "Both cameras must detect the marker; pair not accepted.",
                            file=sys.stderr,
                        )
                        continue
                    accepted = detection_0, detection_1
                    break
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        cv2.destroyAllWindows()

    assert accepted is not None
    try:
        C0_T_target, mean_0, max_0 = _solve_target_pose(
            accepted[0], intrinsics_0.K, intrinsics_0.d
        )
        C1_T_target, mean_1, max_1 = _solve_target_pose(
            accepted[1], intrinsics_1.K, intrinsics_1.d
        )
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    target_dict = target_config.to_target_dict()
    common_notes = "Concurrent stereo one-shot; target held still by operator."
    records = (
        ExtrinsicsRecord(
            W_T_C=se3.inverse(C0_T_target),
            n_frames=1,
            reprojection_error_px_mean=mean_0,
            reprojection_error_px_max=max_0,
            target=target_dict,
            intrinsics_path=str(args.intrinsics_0),
            detection_mode="aruco_marker",
            notes=common_notes,
            camera=camera_block_from_device(args.device_0),
        ),
        ExtrinsicsRecord(
            W_T_C=se3.inverse(C1_T_target),
            n_frames=1,
            reprojection_error_px_mean=mean_1,
            reprojection_error_px_max=max_1,
            target=target_dict,
            intrinsics_path=str(args.intrinsics_1),
            detection_mode="aruco_marker",
            notes=common_notes,
            camera=camera_block_from_device(args.device_1),
        ),
    )
    save_extrinsics(args.out_0, records[0])
    save_extrinsics(args.out_1, records[1])

    C1_T_C0 = relative_camera_transform(C0_T_target, C1_T_target)
    baseline = float(np.linalg.norm(C1_T_C0[:3, 3]))
    print(f"camcal-stereo-extrinsics: wrote {args.out_0} and {args.out_1}")
    print(
        f"C1_T_C0 translation={C1_T_C0[:3, 3].tolist()} m; "
        f"baseline={baseline:.6f} m"
    )
    print(
        f"reprojection camera0 mean/max={mean_0:.3f}/{max_0:.3f} px; "
        f"camera1 mean/max={mean_1:.3f}/{max_1:.3f} px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

