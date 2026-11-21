"""Solve a camera fleet from moving-board observations and write the result.

Two uses:

* full calibration (no ``--fix-pose``): every observed camera's pose, the
  ``--free-intrinsics`` cameras' intrinsics and every board pose are solved;
  the ``world`` capture defines the world;
* one camera was bumped (``--fix-pose`` for the cameras that did not move):
  only the other cameras' poses are solved, intrinsics stay, and the world is
  the one the fixed cameras already define.

``--check`` only reports, with the fleet's current calibration, which camera
moved and whether its intrinsics still fit; it writes no fleet.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import yaml

from . import se3
from .board_bundle import (
    diagnose_rig,
    image_coverage,
    intrinsics_displacement_px,
    solve_board_bundle,
    validate_transfer,
)
from .board_observations import BoardObservationDataset, load_board_observations
from .fleet import CameraFleetRecord, FleetExtrinsics, FleetIntrinsics, load_fleet, save_fleet
from .observations import camera_snapshot_from_fleet

DETECTION_MODE = "charuco_moving_board"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="camcal-board-solve", description=__doc__.splitlines()[0])
    parser.add_argument("--fleet", required=True, type=Path, help="Fleet holding devices and prior calibration")
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Full result and diagnostics (YAML)")
    parser.add_argument("--out-fleet", type=Path, help="Calibrated fleet; not written with --check")
    parser.add_argument("--free-intrinsics", action="append", default=[], metavar="CAMERA")
    parser.add_argument("--fix-pose", action="append", default=[], metavar="CAMERA")
    parser.add_argument("--world-frame", default="board_world", help="World frame name for a full calibration")
    parser.add_argument("--check", action="store_true", help="Only diagnose the current calibration")
    parser.add_argument("--loss", default="huber")
    parser.add_argument("--f-scale-px", type=float, default=2.0)
    parser.add_argument("--max-evaluations", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fleet = load_fleet(args.fleet)
        dataset = _with_fleet_intrinsics(load_board_observations(args.observations), fleet, args.fleet.parent)
        robust = dict(loss=args.loss, f_scale_px=args.f_scale_px)
        report: dict = {"observations": str(args.observations), "fleet": str(args.fleet)}
        current = _current_poses(fleet)
        observed = sorted({obs.camera_name for obs in dataset.observations})
        if args.check or args.fix_pose:
            # Only meaningful while the current calibration is still close to the truth.
            missing = sorted(set(observed) - set(current))
            if missing:
                raise ValueError(f"cameras without a current pose cannot be checked: {missing}")
            report["diagnosis"] = diagnose_rig(dataset, current, **robust)
            print(_diagnosis_text(report["diagnosis"]))
        if args.check:
            _write_yaml(args.out, report)
            print(f"camcal-board-solve: wrote {args.out}")
            return 0

        if args.out_fleet is None:
            raise ValueError("--out-fleet is required unless --check")
        missing = sorted(set(args.fix_pose) - set(current))
        if missing:
            raise ValueError(f"--fix-pose cameras have no current pose: {missing}")
        fixed = {name: current[name] for name in args.fix_pose}
        if args.fix_pose and args.free_intrinsics:
            raise ValueError("re-calibrating a bumped camera keeps intrinsics; drop --free-intrinsics")
        if not fixed:
            stale = sorted(c.name for c in fleet.cameras if c.enabled and c.name not in observed)
            if stale:
                raise ValueError(
                    f"a full calibration moves the world; enabled cameras without observations "
                    f"would keep poses in the old world: {stale}"
                )
        result = solve_board_bundle(
            dataset,
            free_intrinsics=args.free_intrinsics,
            fixed_poses=fixed,
            initial_poses={name: T for name, T in current.items() if name not in fixed} if fixed else None,
            max_evaluations=args.max_evaluations,
            **robust,
        )
        report["solve"] = _solve_report(dataset, result)
        report["validation"] = validate_transfer(dataset, result.K, result.d, result.W_T_C)
        print(_solve_text(report["solve"], report["validation"]))
        calibrated = _calibrated_fleet(fleet, result, world_frame=None if fixed else args.world_frame)
        save_fleet(args.out_fleet, calibrated)
        _write_yaml(args.out, report)
        print(f"camcal-board-solve: wrote {args.out} and {args.out_fleet}")
        return 0 if result.success else 3
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _with_fleet_intrinsics(dataset: BoardObservationDataset, fleet: CameraFleetRecord, fleet_directory: Path):
    """The fleet, not the capture-time snapshot, holds the current intrinsics."""
    cameras = []
    for snapshot in dataset.cameras:
        current = camera_snapshot_from_fleet(fleet.get_camera(snapshot.name), fleet_directory=fleet_directory)
        if current.image_size != snapshot.image_size:
            raise ValueError(
                f"{snapshot.name}: captured at {snapshot.image_size}, the fleet's intrinsics are for {current.image_size}"
            )
        cameras.append(current)
    return replace(dataset, cameras=tuple(cameras))


def _current_poses(fleet: CameraFleetRecord) -> dict[str, np.ndarray]:
    return {
        camera.name: np.asarray(camera.extrinsics.W_T_C, dtype=np.float64)
        for camera in fleet.cameras
        if camera.enabled and camera.extrinsics is not None and camera.extrinsics.W_T_C is not None
    }


def _solve_report(dataset: BoardObservationDataset, result) -> dict:
    snapshots = {camera.name: camera for camera in dataset.cameras}
    left_out = {(view.camera_name, view.capture_id) for view in result.dropped_views}
    cameras = {}
    for name in sorted(result.W_T_C):
        seen = [
            obs
            for obs in dataset.observations
            if obs.camera_name == name and obs.capture_id in result.W_T_B and (name, obs.capture_id) not in left_out
        ]
        points = np.concatenate([obs.image_points for obs in seen]) if seen else np.zeros((0, 2))
        quaternion, translation = se3.to_quat_t(result.W_T_C[name])
        entry = {
            "views": len(seen),
            "coverage": image_coverage(points, snapshots[name].image_size),
            "residual": result.camera_stats[name].as_dict(),
            "W_T_C": result.W_T_C[name].tolist(),
            "translation_m": translation.tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "intrinsics_free": name in result.free_intrinsics,
            "K": result.K[name].tolist(),
            "d": result.d[name].tolist(),
        }
        if name in result.free_intrinsics:
            entry["intrinsics_initialised_from"] = result.initial_intrinsics[name]
            if name in result.zhang_rmse_px:
                entry["zhang_rmse_px"] = result.zhang_rmse_px[name]
            entry["shift_from_prior_px"] = intrinsics_displacement_px(
                snapshots[name].K,
                snapshots[name].d,
                result.K[name],
                result.d[name],
                snapshots[name].image_size,
                observed_points=points,
            )
        cameras[name] = entry
    return {
        "gauge": result.gauge,
        "success": result.success,
        "message": result.message,
        "evaluations": result.evaluations,
        "captures": len(result.W_T_B),
        "dropped_views": [view.as_dict() for view in result.dropped_views],
        "cameras": cameras,
        "board_poses": {capture: T.tolist() for capture, T in sorted(result.W_T_B.items())},
        "capture_rmse_px": {f"{camera}/{capture}": value for (camera, capture), value in sorted(result.observation_rmse_px.items())},
    }


def _calibrated_fleet(fleet: CameraFleetRecord, result, *, world_frame: str | None) -> CameraFleetRecord:
    stamp = datetime.now().astimezone().strftime("%Y%m%d")
    cameras = []
    for camera in fleet.cameras:
        if camera.name not in result.W_T_C:
            cameras.append(camera)
            continue
        stats = result.camera_stats[camera.name]
        intrinsics = camera.intrinsics
        if camera.name in result.free_intrinsics:
            intrinsics = FleetIntrinsics(
                source=f"charuco_moving_board_{stamp}",
                image_size=intrinsics.image_size if intrinsics else None,
                K=result.K[camera.name],
                d=result.d[camera.name],
                camera=intrinsics.camera if intrinsics else None,
                profile=intrinsics.profile if intrinsics else None,
                notes="Estimated jointly with the extrinsics in the moving-board bundle adjustment.",
            )
        extrinsics = FleetExtrinsics(
            detection_mode=DETECTION_MODE,
            W_T_C=result.W_T_C[camera.name],
            reprojection_error_px_mean=stats.mean_px,
            reprojection_error_px_max=stats.max_px,
            notes=f"moving ChArUco board bundle; gauge: {result.gauge}",
        )
        cameras.append(replace(camera, intrinsics=intrinsics, extrinsics=extrinsics))
    return replace(fleet, cameras=cameras, world_frame=world_frame or fleet.world_frame)


def _dropped_text(views: list[dict]) -> list[str]:
    return [
        f"  left out {view['camera']} {view['capture']}: its {view['corners']} corners do not fix the board pose"
        for view in views
    ]


def _diagnosis_text(diagnosis: dict) -> str:
    lines = ["", "Current calibration against these captures (board poses solved, cameras fixed):"]
    lines += _dropped_text(diagnosis["dropped_views"])
    for name, stats in sorted(diagnosis["baseline"].items()):
        lines.append(f"  {name:<14} rmse {stats['rmse_px']:6.2f} px  max {stats['max_px']:6.2f} px")
    lines.append(f"  all cameras    rmse {diagnosis['baseline_rmse_px']:6.2f} px")
    if diagnosis["freed"]:
        lines.append("Freeing one camera's pose at a time (the moved camera brings everything down):")
        for name, entry in sorted(diagnosis["freed"].items()):
            if "error" in entry:
                lines.append(f"  {name:<14} {entry['error']}")
                continue
            lines.append(
                f"  {name:<14} all-camera rmse {entry['rmse_px']:6.2f} px, its pose moves "
                f"{entry['translation_change_mm']:6.1f} mm / {entry['rotation_change_deg']:5.2f} deg"
            )
    return "\n".join(lines)


def _solve_text(solve: dict, validation: list[dict]) -> str:
    lines = ["", f"Bundle adjustment ({solve['gauge']}): {solve['message']}"]
    lines += _dropped_text(solve["dropped_views"])
    for name, entry in solve["cameras"].items():
        residual = entry["residual"]
        line = (
            f"  {name:<14} {entry['views']:3d} views, coverage {entry['coverage']:4.0%}, "
            f"rmse {residual['rmse_px']:5.2f} px, max {residual['max_px']:6.2f} px"
        )
        if "shift_from_prior_px" in entry:
            shift = entry["shift_from_prior_px"]
            line += f" | intrinsics moved {shift['median_px']:.2f} px median, {shift['max_px']:.2f} px max vs prior"
        lines.append(line)
    if validation:
        lines.append("Validation captures (board pose from the other cameras, projected into this one):")
        for row in validation:
            lines.append(
                f"  {row['capture']} {row['camera']:<14} rmse {row['rmse_px']:5.2f} px "
                f"({row['rmse_mm']:5.1f} mm at {row['distance_m']:.2f} m)"
            )
    else:
        lines.append("No validation captures.")
    return "\n".join(lines)


def _write_yaml(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
