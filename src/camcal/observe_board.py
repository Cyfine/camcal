"""Capture a moving ChArUco board with a fixed camera fleet.

Hold the board still and press a key; every camera's current frame is one
capture of that board pose. Each tile shows the corners detected now and, as
dots, every corner captured so far, so gaps in a camera's image are visible.

Keys: SPACE solve capture, W world capture (board flat on the floor), V
validation capture, U undo the last capture, ENTER save, ESC or q abort.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import sys

import cv2
import numpy as np

from .board_bundle import image_coverage
from .board_observations import (
    BoardCapture,
    BoardObservation,
    BoardObservationDataset,
    save_board_observations,
)
from .charuco import CharucoBoard, CharucoDetection
from .fleet import load_fleet
from .inspect_cameras import compose_preview_grid, fit_frame_to_tile
from .io_yaml import load_board
from .observations import camera_snapshot_from_fleet
from .observe_fleet import _camera_fps, _capture_camera, _draw_canvas_status, _select_fleet_cameras

WINDOW_TITLE = "camcal: moving ChArUco board"
ROLE_COLOURS = {"solve": (0, 200, 255), "world": (255, 0, 255), "validation": (255, 200, 0)}
KEY_ROLES = {ord(" "): "solve", ord("w"): "world", ord("v"): "validation"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="camcal-board-observe", description=__doc__.splitlines()[0])
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--tile-width", type=int, default=800)
    parser.add_argument("--tile-height", type=int, default=450)
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--notes", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = observe_live(
            fleet_path=args.fleet,
            board_path=args.board,
            output_path=args.out,
            camera_names=args.camera or None,
            tile_size=(args.tile_width, args.tile_height),
            columns=args.columns or None,
            notes=args.notes,
        )
        if dataset is None:
            print("camcal-board-observe: aborted", file=sys.stderr)
            return 1
        save_board_observations(args.out, dataset)
        print(f"camcal-board-observe: wrote {args.out}")
        print(describe(dataset))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def observe_live(
    *,
    fleet_path: Path,
    board_path: Path,
    output_path: Path,
    camera_names: list[str] | None,
    tile_size: tuple[int, int],
    columns: int | None,
    notes: str,
) -> BoardObservationDataset | None:
    fleet = load_fleet(fleet_path)
    board_config = load_board(board_path)
    board = CharucoBoard.from_config(board_config)
    fleet_cameras = _select_fleet_cameras(fleet, camera_names)
    snapshots = tuple(camera_snapshot_from_fleet(camera, fleet_directory=fleet_path.parent) for camera in fleet_cameras)
    image_directory = output_path.parent / f"{output_path.stem}_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    captures: list[BoardCapture] = []
    observations: list[BoardObservation] = []
    cameras = {}
    try:
        # RealSense first: opening it after the UVC cameras can starve its USB bandwidth.
        order = sorted(zip(fleet_cameras, snapshots), key=lambda pair: pair[0].device.get("backend") != "realsense")
        for fleet_camera, snapshot in order:
            camera = _capture_camera(fleet_camera, snapshot.image_size, _camera_fps(snapshot.profile))
            camera.connect()
            if camera.image_size() != snapshot.image_size:
                raise ValueError(
                    f"camera {snapshot.name!r} streams at {camera.image_size()}, "
                    f"intrinsics expect {snapshot.image_size}"
                )
            cameras[snapshot.name] = camera

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        next_index = 0
        while True:
            frames = {snapshot.name: cameras[snapshot.name].capture() for snapshot in snapshots}
            detections = {name: board.detect_full(frame) for name, frame in frames.items()}
            dataset = _dataset(board_config, snapshots, captures, observations, notes)
            tiles = [
                fit_frame_to_tile(
                    _draw_tile(frames[s.name], detections[s.name], s, dataset),
                    tile_width=tile_size[0],
                    tile_height=tile_size[1],
                )
                for s in snapshots
            ]
            canvas = compose_preview_grid(tiles, columns=columns)
            _draw_canvas_status(
                canvas,
                f"{describe(dataset)} | SPACE solve, W world, V validation, U undo, ENTER save, ESC abort",
            )
            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return None
            if key == 13:
                if captures:
                    return _dataset(board_config, snapshots, captures, observations, notes, final=True)
                continue
            if key == ord("u") and captures:
                removed = captures.pop()
                observations = [obs for obs in observations if obs.capture_id != removed.capture_id]
                for path in image_directory.glob(f"{removed.capture_id}_*.png"):
                    path.unlink()
                continue
            role = KEY_ROLES.get(key)
            if role is None:
                continue
            seen = {name: detection for name, detection in detections.items() if detection is not None}
            if not seen:
                continue
            if role == "world":
                captures = [BoardCapture(c.capture_id, "solve", c.captured_at) if c.role == "world" else c for c in captures]
            capture_id = f"cap-{next_index:04d}"
            next_index += 1
            captures.append(BoardCapture(capture_id, role, _utc_now_iso()))
            for name, detection in seen.items():
                image_path = image_directory / f"{capture_id}_{name}.png"
                cv2.imwrite(str(image_path), frames[name])
                observations.append(
                    BoardObservation(
                        name,
                        capture_id,
                        detection.corner_ids,
                        detection.image_points,
                        source_image=str(image_path.relative_to(output_path.parent)),
                    )
                )
    finally:
        cv2.destroyAllWindows()
        for camera in cameras.values():
            camera.disconnect()


def describe(dataset: BoardObservationDataset) -> str:
    roles = [capture.role for capture in dataset.captures]
    by_capture: dict[str, set[str]] = {}
    for obs in dataset.observations:
        by_capture.setdefault(obs.capture_id, set()).add(obs.camera_name)
    names = [camera.name for camera in dataset.cameras]
    pairs = []
    for first, second in combinations(names, 2):
        count = sum(1 for seen in by_capture.values() if {first, second} <= seen)
        pairs.append(f"{first}+{second} {count}")
    world = "world set" if "world" in roles else "NO world"
    return (
        f"{roles.count('solve')} solve, {roles.count('validation')} validation, {world} | "
        f"shared: {', '.join(pairs)}"
    )


def _dataset(board_config, snapshots, captures, observations, notes, final=False):
    return BoardObservationDataset(
        board=board_config,
        cameras=snapshots,
        captures=tuple(captures),
        observations=tuple(observations),
        created_at=_utc_now_iso() if final else "",
        notes=notes,
    )


def _draw_tile(frame: np.ndarray, detection: CharucoDetection | None, snapshot, dataset) -> np.ndarray:
    shown = frame.copy()
    roles = {capture.capture_id: capture.role for capture in dataset.captures}
    captured = [obs for obs in dataset.observations if obs.camera_name == snapshot.name]
    for obs in captured:
        colour = ROLE_COLOURS[roles[obs.capture_id]]
        for x, y in np.rint(obs.image_points).astype(int):
            cv2.circle(shown, (int(x), int(y)), 6, colour, -1, cv2.LINE_AA)
    corners = 0
    if detection is not None:
        corners = len(detection.corner_ids)
        for x, y in np.rint(detection.image_points).astype(int):
            cv2.circle(shown, (int(x), int(y)), 10, (0, 230, 0), 3, cv2.LINE_AA)
    points = np.concatenate([obs.image_points for obs in captured]) if captured else np.zeros((0, 2))
    coverage = image_coverage(points, snapshot.image_size)
    text = f"{snapshot.name}: {corners} corners now | {len(captured)} views | coverage {coverage:.0%}"
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(shown, text, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return shown


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


if __name__ == "__main__":
    raise SystemExit(main())
