from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from camcal import se3
from camcal.charuco import CharucoBoard
from camcal.io_yaml import (
    ExtrinsicsRecord,
    RealSenseIntrinsicsCameraRecord,
    load_board,
    load_realsense_intrinsics_bundle,
    profile_image_size,
    save_extrinsics,
)
from camcal.pnp import reprojection_error_px, solve_pnp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture RealSense color frames and solve board-frame extrinsics."
    )
    parser.add_argument("--board", type=Path, default=Path("board.yaml"))
    parser.add_argument(
        "--intrinsics", type=Path, default=Path("realsense_intrinsics.yaml")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("realsense_extrinsics"))
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = args.out_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    board_cfg = load_board(args.board)
    board = CharucoBoard.from_config(board_cfg)
    bundle = load_realsense_intrinsics_bundle(args.intrinsics)
    profile = bundle.profile
    width, height = profile_image_size(profile)
    fps = int(profile["fps"])

    ok_count = 0
    for cam in bundle.cameras:
        serial = cam.serial
        ok = _solve_one_camera(
            serial=serial,
            cam=cam,
            profile=profile,
            board=board,
            board_cfg=board_cfg,
            intrinsics_path=args.intrinsics,
            out_dir=args.out_dir,
            capture_dir=capture_dir,
            width=int(width),
            height=int(height),
            fps=fps,
            num_frames=args.num_frames,
            timeout_s=args.timeout_s,
        )
        ok_count += int(ok)

    return 0 if ok_count else 2


def _solve_one_camera(
    *,
    serial: str,
    cam: RealSenseIntrinsicsCameraRecord,
    profile: dict[str, object],
    board: CharucoBoard,
    board_cfg: object,
    intrinsics_path: Path,
    out_dir: Path,
    capture_dir: Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    timeout_s: float,
) -> bool:
    K = np.asarray(cam.K, dtype=np.float64)
    d = np.asarray(cam.d, dtype=np.float64)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    print(f"[{serial}] starting {width}x{height}@{fps} color")
    pipe.start(cfg)

    detections = []
    frames_seen = 0
    last_frame = None
    start = time.monotonic()
    try:
        while time.monotonic() - start < timeout_s and len(detections) < num_frames:
            frames = pipe.wait_for_frames(3000)
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            last_frame = frame.copy()
            frames_seen += 1
            if frames_seen < 20:
                continue
            det = board.detect_full(frame)
            if det is None:
                continue
            detections.append(det)
            print(f"[{serial}] detection {len(detections)}: {len(det.corner_ids)} corners")
            time.sleep(0.15)
    finally:
        pipe.stop()

    if last_frame is not None:
        cv2.imwrite(str(capture_dir / f"{serial}_last.png"), last_frame)

    if not detections:
        print(f"[{serial}] no board detections in {frames_seen} frames")
        return False

    per_frame_W_T_C = []
    means = []
    maxes = []
    for det in detections:
        C_T_W = solve_pnp(det.object_points, det.image_points, K, d)
        if C_T_W is None:
            continue
        W_T_C = se3.inverse(C_T_W)
        mean_px, max_px = reprojection_error_px(
            object_points=det.object_points,
            image_points=det.image_points,
            K=K,
            d=d,
            C_T_T=C_T_W,
        )
        per_frame_W_T_C.append(W_T_C)
        means.append(mean_px)
        maxes.append(max_px)

    if not per_frame_W_T_C:
        print(f"[{serial}] detections found but PnP failed")
        return False

    W_T_C = (
        per_frame_W_T_C[0]
        if len(per_frame_W_T_C) == 1
        else se3.mean_se3(per_frame_W_T_C)
    )
    out_path = out_dir / f"{serial}_extrinsics.yaml"
    record = ExtrinsicsRecord(
        W_T_C=W_T_C,
        board=board_cfg,
        n_frames=len(per_frame_W_T_C),
        reprojection_error_px_mean=float(np.mean(means)),
        reprojection_error_px_max=float(np.max(maxes)),
        intrinsics_path=str(intrinsics_path),
        notes=(
            f"RealSense factory intrinsics, serial {serial}, "
            f"{width}x{height}@{fps} {profile['stream']}/{profile['format']}"
        ),
        camera={
            "serial": serial,
            "model": cam.model,
            "product_id": cam.product_id,
            "firmware_version": cam.firmware_version,
            "source": "realsense_api",
            "stream": profile["stream"],
            "format": profile["format"],
            "fps": fps,
            "image_size": [width, height],
        },
        per_frame_W_T_C=per_frame_W_T_C if len(per_frame_W_T_C) > 1 else [],
    )
    save_extrinsics(out_path, record)
    t = W_T_C[:3, 3]
    print(
        f"[{serial}] wrote {out_path}: "
        f"t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m "
        f"mean={record.reprojection_error_px_mean:.3f}px "
        f"max={record.reprojection_error_px_max:.3f}px "
        f"frames={record.n_frames}"
    )
    return True


if __name__ == "__main__":
    raise SystemExit(main())
