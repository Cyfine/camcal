"""Capture a moving ChArUco board with a fixed camera fleet.

Hold the board still; every camera's current frame is one capture of that
board pose. Each tile shows the corners detected now and, as dots, every
corner captured so far, so gaps in a camera's image are visible.

Manual keys: SPACE solve capture, W world capture (board face up on the
floor), V validation capture, U undo the last capture, ENTER save, ESC or q
abort.

Auto capture (``--auto`` or key A) lets one person calibrate alone. A capture
is taken when the board has been steady in every camera that sees it, and only
when it shows at least one camera a view unlike its earlier ones: the board
centre elsewhere in the image, a different tilt, or a different distance.
Holding the board in one place therefore yields one capture, not a stream of
duplicates. A shutter sound and a flash of the window border mark each
capture. In auto mode W arms the next capture as the world capture (it need
not be a new view), and V switches later captures between solve and
validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from .board_bundle import _pnp, image_coverage
from .board_observations import (
    BoardCapture,
    BoardObservation,
    BoardObservationDataset,
    board_corner_points,
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
SHUTTER_SOUND = Path("/usr/share/sounds/freedesktop/stereo/camera-shutter.oga")
FLASH_S = 0.4


@dataclass(frozen=True)
class AutoSettings:
    """When a still board is captured automatically."""

    # Steady: corners slower than still_px_per_s for still_s. The cameras are not
    # synchronised; their frames can be up to one frame period (33 ms) apart, so
    # 15 px/s costs at most about 0.5 px of disagreement between cameras.
    still_s: float = 0.3
    still_px_per_s: float = 15.0
    new_centre_fraction: float = 0.15
    new_tilt_deg: float = 15.0
    new_distance_fraction: float = 0.2


@dataclass(frozen=True)
class View:
    """Where the board is for one camera: image centre (px), rotation, distance (m)."""

    centre_px: np.ndarray
    rotation: np.ndarray
    distance_m: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="camcal-board-observe", description=__doc__.splitlines()[0])
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--auto", action="store_true", help="Start with auto capture on")
    defaults = AutoSettings()
    parser.add_argument("--still-s", type=float, default=defaults.still_s)
    parser.add_argument("--still-px-per-s", type=float, default=defaults.still_px_per_s)
    parser.add_argument("--new-centre-fraction", type=float, default=defaults.new_centre_fraction)
    parser.add_argument("--new-tilt-deg", type=float, default=defaults.new_tilt_deg)
    parser.add_argument("--new-distance-fraction", type=float, default=defaults.new_distance_fraction)
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
            auto=args.auto,
            settings=AutoSettings(
                args.still_s,
                args.still_px_per_s,
                args.new_centre_fraction,
                args.new_tilt_deg,
                args.new_distance_fraction,
            ),
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


def view_of(detection: CharucoDetection, K: np.ndarray, d: np.ndarray, corners: np.ndarray, centre_b: np.ndarray) -> View:
    C_T_B = _pnp(K, d, corners[detection.corner_ids], detection.image_points)
    centre_c = C_T_B[:3, :3] @ centre_b + C_T_B[:3, 3]
    pixel, _ = cv2.projectPoints(centre_c.reshape(1, 3), np.zeros(3), np.zeros(3), K, d)
    return View(pixel.reshape(2), C_T_B[:3, :3].copy(), float(np.linalg.norm(centre_c)))


def is_new_view(view: View, earlier: list[View], image_width: int, settings: AutoSettings) -> bool:
    """New when, against every earlier view, the centre, tilt or distance differs enough."""
    for other in earlier:
        moved = np.linalg.norm(view.centre_px - other.centre_px) > settings.new_centre_fraction * image_width
        cosine = (np.trace(view.rotation.T @ other.rotation) - 1.0) / 2.0
        turned = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > settings.new_tilt_deg
        scaled = abs(np.log(view.distance_m / other.distance_m)) > np.log1p(settings.new_distance_fraction)
        if not (moved or turned or scaled):
            return False
    return True


def corner_motion_px(previous: CharucoDetection | None, current: CharucoDetection) -> float:
    """Median movement of the corners seen in both frames; infinite when none are shared."""
    if previous is None:
        return float("inf")
    shared, now, before = np.intersect1d(current.corner_ids, previous.corner_ids, return_indices=True)
    if shared.size < 4:
        return float("inf")
    return float(np.median(np.linalg.norm(current.image_points[now] - previous.image_points[before], axis=1)))


def observe_live(
    *,
    fleet_path: Path,
    board_path: Path,
    output_path: Path,
    camera_names: list[str] | None,
    tile_size: tuple[int, int],
    columns: int | None,
    notes: str,
    auto: bool = False,
    settings: AutoSettings = AutoSettings(),
) -> BoardObservationDataset | None:
    fleet = load_fleet(fleet_path)
    board_config = load_board(board_path)
    board = CharucoBoard.from_config(board_config)
    corners = board_corner_points(board_config)
    centre_b = np.array([board_config.squares_x, board_config.squares_y, 0.0]) * board_config.square_size_m / 2
    fleet_cameras = _select_fleet_cameras(fleet, camera_names)
    snapshots = tuple(camera_snapshot_from_fleet(camera, fleet_directory=fleet_path.parent) for camera in fleet_cameras)
    by_name = {snapshot.name: snapshot for snapshot in snapshots}
    image_directory = output_path.parent / f"{output_path.stem}_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    captures: list[BoardCapture] = []
    observations: list[BoardObservation] = []
    views: dict[str, list[tuple[str, View]]] = {snapshot.name: [] for snapshot in snapshots}
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
        previous: dict[str, CharucoDetection | None] = {name: None for name in by_name}
        still_since: float | None = None
        # World arming: None, "waiting" (for the board to move or vanish first), "ready".
        auto_role, world_armed = "solve", None
        flash_until, message = 0.0, ""
        previous_time = time.monotonic()
        grabber = ThreadPoolExecutor(max_workers=len(cameras))
        while True:
            # Grab every camera at once, then decode: frames as close in time as possible.
            list(grabber.map(lambda name: cameras[name].grab(), by_name))
            now = time.monotonic()
            frames = {name: cameras[name].retrieve() for name in by_name}
            detections = {name: board.detect_full(frame) for name, frame in frames.items()}
            seen = {name: detection for name, detection in detections.items() if detection is not None}
            limit_px = settings.still_px_per_s * max(now - previous_time, 1e-3)
            moving = [name for name, detection in seen.items() if corner_motion_px(previous[name], detection) > limit_px]
            previous, previous_time = detections, now
            still_since = None if (not seen or moving) else (still_since or now)
            still_for = 0.0 if still_since is None else now - still_since
            if world_armed == "waiting" and (not seen or moving):
                world_armed = "ready"

            role = None
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return None
            if key == 13:
                if captures:
                    return _dataset(board_config, snapshots, captures, observations, notes, final=True)
            elif key == ord("a"):
                auto, world_armed = not auto, None
            elif key == ord("u") and captures:
                removed = captures.pop()
                observations = [obs for obs in observations if obs.capture_id != removed.capture_id]
                for name in views:
                    views[name] = [(capture, view) for capture, view in views[name] if capture != removed.capture_id]
                for path in image_directory.glob(f"{removed.capture_id}_*.png"):
                    path.unlink()
                message = f"undid {removed.capture_id}"
            elif auto and key == ord("w"):
                # Wait for the board to move first, so the pose held now is not taken as the world.
                world_armed = "waiting"
            elif auto and key == ord("v"):
                auto_role = "validation" if auto_role == "solve" else "solve"
            elif not auto and key in KEY_ROLES and seen:
                role = KEY_ROLES[key]

            reason = ""
            candidate = {}
            if seen and (role is not None or (auto and still_for >= settings.still_s)):
                for name, detection in seen.items():
                    try:
                        candidate[name] = view_of(detection, by_name[name].K, by_name[name].d, corners, centre_b)
                    except (ValueError, cv2.error):
                        continue
            if role is None and auto and candidate:
                if world_armed == "ready":
                    role = "world"
                elif world_armed is None:
                    new = [
                        name
                        for name, view in candidate.items()
                        if is_new_view(view, [v for _, v in views[name]], by_name[name].image_size[0], settings)
                    ]
                    if new:
                        role = auto_role
                        reason = " (new view: " + ", ".join(new) + ")"

            if role is not None and not candidate:
                role, message = None, "no view fixes the board pose: show more of the board"
            if role is not None:
                if role == "world":
                    captures = [BoardCapture(c.capture_id, "solve", c.captured_at) if c.role == "world" else c for c in captures]
                    world_armed = None
                capture_id = f"cap-{next_index:04d}"
                next_index += 1
                captures.append(BoardCapture(capture_id, role, _utc_now_iso()))
                # Only views whose corners fix the board pose: the solve starts from each one's pose.
                for name, view in candidate.items():
                    detection = seen[name]
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
                    views[name].append((capture_id, view))
                flash_until = now + FLASH_S
                _shutter()
                message = f"{capture_id} {role}{reason}"

            dataset = _dataset(board_config, snapshots, captures, observations, notes)
            tiles = [
                fit_frame_to_tile(
                    _draw_tile(frames[s.name], detections[s.name], s, dataset, s.name in moving),
                    tile_width=tile_size[0],
                    tile_height=tile_size[1],
                )
                for s in snapshots
            ]
            canvas = compose_preview_grid(tiles, columns=columns)
            if now < flash_until:
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (255, 255, 255), 24)
            if auto:
                target = {"waiting": "WORLD next: move the board", "ready": "WORLD next"}.get(world_armed, auto_role)
                mode = f"AUTO {target} | still {min(still_for, settings.still_s):.1f}/{settings.still_s:.1f}s"
                keys = "A manual, W world next, V solve/validation, U undo, ENTER save, ESC abort"
            else:
                mode = "MANUAL"
                keys = "SPACE solve, W world, V validation, A auto, U undo, ENTER save, ESC abort"
            _draw_canvas_status(canvas, f"{mode} | {describe(dataset)} | {message} | {keys}")
            cv2.imshow(WINDOW_TITLE, canvas)
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


def _shutter() -> None:
    """Audible capture feedback for an operator away from the screen."""
    for player in ("pw-play", "canberra-gtk-play", "paplay"):
        path = shutil.which(player)
        if path and SHUTTER_SOUND.exists():
            arguments = [path, "-f", str(SHUTTER_SOUND)] if player == "canberra-gtk-play" else [path, str(SHUTTER_SOUND)]
            subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    sys.stdout.write("\a")
    sys.stdout.flush()


def _dataset(board_config, snapshots, captures, observations, notes, final=False):
    return BoardObservationDataset(
        board=board_config,
        cameras=snapshots,
        captures=tuple(captures),
        observations=tuple(observations),
        created_at=_utc_now_iso() if final else "",
        notes=notes,
    )


def _draw_tile(frame: np.ndarray, detection: CharucoDetection | None, snapshot, dataset, moving: bool) -> np.ndarray:
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
        colour = (0, 165, 255) if moving else (0, 230, 0)
        for x, y in np.rint(detection.image_points).astype(int):
            cv2.circle(shown, (int(x), int(y)), 10, colour, 3, cv2.LINE_AA)
    points = np.concatenate([obs.image_points for obs in captured]) if captured else np.zeros((0, 2))
    coverage = image_coverage(points, snapshot.image_size)
    state = " moving" if moving else ""
    text = f"{snapshot.name}: {corners} corners{state} | {len(captured)} views | coverage {coverage:.0%}"
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(shown, text, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return shown


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


if __name__ == "__main__":
    raise SystemExit(main())
