"""Solve and inspect one common world frame for a static camera fleet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

from . import se3
from .fleet_bundle import (
    RoomStructure,
    derive_table_frame,
    save_calibrated_fleet,
    save_fleet_bundle_result,
    solve_fleet_bundle,
)
from .fleet_visualization import (
    write_fleet_calibration_html,
    write_residual_overlays,
)
from .marker_map import load_marker_manifest
from .observations import load_observation_dataset, merge_observation_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camcal-fleet-solve",
        description=(
            "Jointly solve static camera and marker poses; world is the configured "
            "anchor marker center/orientation."
        ),
    )
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--markers", required=True, type=Path)
    parser.add_argument("--observations", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-fleet", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--residual-overlays", type=Path, default=None)
    parser.add_argument(
        "--room-anchor-marker-id",
        type=int,
        required=True,
        help="Room-fixed marker whose center/orientation defines the BA gauge.",
    )
    parser.add_argument(
        "--table-marker-id",
        type=int,
        help=(
            "Marker rigidly attached to the table. Use together with "
            "--table-marker-pose."
        ),
    )
    parser.add_argument(
        "--table-marker-pose",
        type=Path,
        help=(
            "YAML containing the known 4x4 T_T_M pose of the table marker. "
            "Use together with --table-marker-id."
        ),
    )
    parser.add_argument(
        "--room-structure",
        type=Path,
        help=(
            "Optional YAML declaring floor_marker_id and "
            "parallel_wall_marker_groups."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("linear", "soft_l1", "huber", "cauchy"),
        default="soft_l1",
        help="Robust least-squares loss; raw residuals are always retained.",
    )
    parser.add_argument("--f-scale-px", type=float, default=2.0)
    parser.add_argument("--max-evaluations", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_marker_manifest(args.markers)
        datasets = [load_observation_dataset(path) for path in args.observations]
        dataset = merge_observation_datasets(datasets)
        if (args.table_marker_id is None) != (args.table_marker_pose is None):
            raise ValueError(
                "--table-marker-id and --table-marker-pose must be supplied together"
            )
        result = solve_fleet_bundle(
            dataset,
            manifest,
            anchor_marker_id=args.room_anchor_marker_id,
            room_structure=(
                _load_room_structure(args.room_structure)
                if args.room_structure is not None
                else None
            ),
            loss=args.loss,
            f_scale_px=args.f_scale_px,
            max_nfev=args.max_evaluations,
        )
        table_frame = (
            derive_table_frame(
                result,
                table_marker_id=args.table_marker_id,
                table_T_marker=_load_table_T_marker(args.table_marker_pose),
            )
            if args.table_marker_id is not None
            else None
        )
        out_fleet = args.out_fleet or args.out.with_name(
            f"{args.out.stem}_fleet.yaml"
        )
        html = args.html or args.out.with_suffix(".html")
        overlays = args.residual_overlays or args.out.with_name(
            f"{args.out.stem}_residual_overlays"
        )
        save_fleet_bundle_result(
            args.out,
            result,
            observation_paths=args.observations,
            marker_manifest_path=args.markers,
            table_frame=table_frame,
        )
        write_fleet_calibration_html(
            html,
            result,
            display_T_world=(
                table_frame.table_T_room if table_frame is not None else None
            ),
            display_frame_name="cad_table" if table_frame is not None else None,
        )
        overlay_paths = write_residual_overlays(
            overlays,
            result,
            dataset,
            manifest,
        )
        print(
            f"camcal-fleet-solve: "
            f"{'SOLVED' if result.success else 'FAILED TO SOLVE'} — "
            f"global RMSE {result.global_residual.rmse_px:.3f} px"
        )
        print(
            f"BA anchor = marker {result.anchor_marker_id}; A_T_C maps camera "
            "coordinates into the anchor-marker frame"
        )
        if table_frame is not None:
            print(
                "CAD table export enabled; T_T_C maps camera coordinates into "
                "the floor-base table frame"
            )
        print(f"result: {args.out}")
        print(f"visual review: {html}")
        if overlay_paths:
            print(f"residual overlays: {overlays} ({len(overlay_paths)} images)")
        if result.success:
            save_calibrated_fleet(
                args.fleet,
                out_fleet,
                result,
                table_frame=table_frame,
            )
            print(f"calibrated fleet: {out_fleet}")
            print("accuracy acceptance is not automatic; review residuals and 3D layout")
        else:
            print("no calibrated fleet was written from the failed solve")
        return 0 if result.success else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _load_table_T_marker(path: Path) -> np.ndarray:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "T_T_M" not in raw:
        raise ValueError(f"{path}: expected a YAML mapping containing T_T_M")
    pose = np.asarray(raw["T_T_M"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{path}: T_T_M must be 4x4, got {pose.shape}")
    se3.assert_se3(pose)
    return pose


def _load_room_structure(path: Path) -> RoomStructure:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: room structure must be a YAML mapping")
    try:
        return RoomStructure(
            floor_marker_id=int(raw["floor_marker_id"]),
            parallel_wall_marker_groups=tuple(
                tuple(int(marker_id) for marker_id in group)
                for group in raw["parallel_wall_marker_groups"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid room structure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
