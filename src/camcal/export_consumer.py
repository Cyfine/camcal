"""Export fleet camera state into the consumer's 2-camera calibration bundle.

The current consumer expects a calibration directory rooted at a `calib_path`
with four files:

* `c0_params_color.yaml`
* `c2_params_color.yaml`
* `c0_to_c2_params_color.yaml`
* `camera0_pose.yaml`

This module exports those artifacts from the stateful fleet config so the
consumer receives the exact OpenCV/YAML formats it already knows how to load.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import se3
from .fleet import CameraFleetRecord, FleetCamera, load_fleet
from .io_yaml import (
    load_extrinsics,
    resolve_intrinsics_file,
)


@dataclass(frozen=True)
class ConsumerBundlePaths:
    """Output paths for one exported consumer calibration bundle."""

    c0_intrinsics: Path
    c2_intrinsics: Path
    c0_to_c2_extrinsics: Path
    camera0_pose: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-export-consumer",
        description=(
            "Export two named cameras from a fleet config into the consumer's "
            "c0/c2 calibration bundle."
        ),
    )
    p.add_argument(
        "--fleet",
        required=True,
        type=Path,
        help="Path to fleet.yaml.",
    )
    p.add_argument(
        "--camera0",
        required=True,
        help="Fleet camera name that should become consumer camera0 / c0.",
    )
    p.add_argument(
        "--camera2",
        required=True,
        help="Fleet camera name that should become consumer camera2 / c2.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where the consumer bundle should be written.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = export_consumer_bundle(
            fleet_path=args.fleet,
            camera0_name=args.camera0,
            camera2_name=args.camera2,
            out_dir=args.out_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "camcal-export-consumer: wrote "
        f"{paths.c0_intrinsics}, {paths.c2_intrinsics}, "
        f"{paths.c0_to_c2_extrinsics}, {paths.camera0_pose}",
        file=sys.stderr,
    )
    return 0


def export_consumer_bundle(
    *,
    fleet_path: str | Path,
    camera0_name: str,
    camera2_name: str,
    out_dir: str | Path,
) -> ConsumerBundlePaths:
    """Export one consumer 2-camera bundle from a fleet config."""
    fleet_path = Path(fleet_path)
    out_dir = Path(out_dir)
    if camera0_name == camera2_name:
        raise ValueError("--camera0 and --camera2 must refer to different cameras")

    fleet = load_fleet(fleet_path)
    fleet_dir = fleet_path.parent
    cam0 = fleet.get_camera(camera0_name)
    cam2 = fleet.get_camera(camera2_name)

    K0, d0, image_size0 = _resolve_camera_intrinsics(cam0, fleet_dir=fleet_dir)
    K2, d2, image_size2 = _resolve_camera_intrinsics(cam2, fleet_dir=fleet_dir)
    if image_size0 != image_size2:
        raise ValueError(
            "consumer bundle assumes a shared calibration image size; got "
            f"{camera0_name}={image_size0} and {camera2_name}={image_size2}"
        )

    W_T_C0 = _resolve_camera_world_pose(cam0, fleet_dir=fleet_dir)
    W_T_C2 = _resolve_camera_world_pose(cam2, fleet_dir=fleet_dir)
    C2_T_C0 = se3.compose(se3.inverse(W_T_C2), W_T_C0)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = ConsumerBundlePaths(
        c0_intrinsics=out_dir / "c0_params_color.yaml",
        c2_intrinsics=out_dir / "c2_params_color.yaml",
        c0_to_c2_extrinsics=out_dir / "c0_to_c2_params_color.yaml",
        camera0_pose=out_dir / "camera0_pose.yaml",
    )

    _write_opencv_intrinsics(bundle_paths.c0_intrinsics, K=K0, d=d0)
    _write_opencv_intrinsics(bundle_paths.c2_intrinsics, K=K2, d=d2)
    _write_opencv_rt(bundle_paths.c0_to_c2_extrinsics, T=C2_T_C0)
    _write_plain_pose(bundle_paths.camera0_pose, T=W_T_C0)
    return bundle_paths


def _resolve_camera_intrinsics(
    camera: FleetCamera,
    *,
    fleet_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    intrinsics = camera.intrinsics
    if intrinsics is None:
        raise ValueError(f"Fleet camera {camera.name!r} has no intrinsics state")

    if (
        intrinsics.K is not None
        and intrinsics.d is not None
        and intrinsics.image_size is not None
    ):
        K = np.asarray(intrinsics.K, dtype=np.float64)
        d = np.asarray(intrinsics.d, dtype=np.float64).flatten()
        image_size = tuple(int(x) for x in intrinsics.image_size)
        _validate_consumer_intrinsics(camera_name=camera.name, K=K, d=d)
        return K, d, (image_size[0], image_size[1])

    if not intrinsics.path:
        raise ValueError(
            f"Fleet camera {camera.name!r} has intrinsics metadata but neither "
            "inline K/d/image_size nor a path to load from"
        )

    selector_serial = _camera_serial(camera)
    record, _ = resolve_intrinsics_file(
        _resolve_relative_path(fleet_dir, intrinsics.path),
        camera_serial=selector_serial,
    )
    K = np.asarray(record.K, dtype=np.float64)
    d = np.asarray(record.d, dtype=np.float64).flatten()
    _validate_consumer_intrinsics(camera_name=camera.name, K=K, d=d)
    return K, d, record.image_size


def _validate_consumer_intrinsics(
    *,
    camera_name: str,
    K: np.ndarray,
    d: np.ndarray,
) -> None:
    if K.shape != (3, 3):
        raise ValueError(
            f"Fleet camera {camera_name!r} must export a 3x3 K, got {K.shape}"
        )
    if d.size != 5:
        raise ValueError(
            f"Fleet camera {camera_name!r} must export a 5-coefficient OpenCV "
            f"distortion vector for the current consumer, got length {d.size}"
        )


def _resolve_camera_world_pose(
    camera: FleetCamera,
    *,
    fleet_dir: Path,
) -> np.ndarray:
    extrinsics = camera.extrinsics
    if extrinsics is None:
        raise ValueError(f"Fleet camera {camera.name!r} has no extrinsics state")

    if extrinsics.W_T_C is not None:
        W_T_C = np.asarray(extrinsics.W_T_C, dtype=np.float64)
        se3.assert_se3(W_T_C)
        return W_T_C

    if not extrinsics.path:
        raise ValueError(
            f"Fleet camera {camera.name!r} has extrinsics metadata but neither "
            "an inline W_T_C nor a path to load from"
        )

    record = load_extrinsics(_resolve_relative_path(fleet_dir, extrinsics.path))
    return np.asarray(record.W_T_C, dtype=np.float64)


def _camera_serial(camera: FleetCamera) -> str | None:
    if "serial" in camera.device and camera.device["serial"] is not None:
        return str(camera.device["serial"])
    if camera.intrinsics is not None and camera.intrinsics.camera is not None:
        serial = camera.intrinsics.camera.get("serial")
        if serial is not None:
            return str(serial)
    if camera.extrinsics is not None and camera.extrinsics.camera is not None:
        serial = camera.extrinsics.camera.get("serial")
        if serial is not None:
            return str(serial)
    return None


def _resolve_relative_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _write_opencv_intrinsics(path: Path, *, K: np.ndarray, d: np.ndarray) -> None:
    _write_opencv_matrices(
        path,
        {
            "K": np.asarray(K, dtype=np.float64).reshape(3, 3),
            "D": np.asarray(d, dtype=np.float64).reshape(1, 5),
        },
    )


def _write_opencv_rt(path: Path, *, T: np.ndarray) -> None:
    R, t = se3.to_Rt(np.asarray(T, dtype=np.float64))
    _write_opencv_matrices(
        path,
        {
            "R": R.reshape(3, 3),
            "T": t.reshape(3, 1),
        },
    )


def _write_opencv_matrices(path: Path, matrices: dict[str, np.ndarray]) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"could not open {path} for OpenCV FileStorage writing")
    try:
        for key, matrix in matrices.items():
            fs.write(key, np.asarray(matrix, dtype=np.float64))
    finally:
        fs.release()


def _write_plain_pose(path: Path, *, T: np.ndarray) -> None:
    R, t = se3.to_Rt(np.asarray(T, dtype=np.float64))
    body = {
        "rotation_matrix": {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [float(x) for x in R.reshape(-1)],
        },
        "translation_vector": {
            "rows": 3,
            "cols": 1,
            "dt": "d",
            "data": [float(x) for x in t.reshape(-1)],
        },
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False))


__all__ = [
    "ConsumerBundlePaths",
    "build_parser",
    "export_consumer_bundle",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
