"""Capture or import static multi-camera ArUco observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
from pathlib import Path
import socket
import sys

import cv2
import numpy as np

from .aruco import ArucoMarkerDetection, ArucoMultiMarkerDetector
from .camera import Camera, RealSenseCamera
from .fleet import FleetCamera, load_fleet
from .inspect_cameras import compose_preview_grid, fit_frame_to_tile
from .marker_map import load_marker_manifest
from .observation_graph import build_observation_graph
from .observations import (
    MarkerObservation,
    ObservationDataset,
    camera_snapshot_from_fleet,
    save_observation_dataset,
)


WINDOW_TITLE = "camcal: fleet marker observations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camcal-fleet-observe",
        description=(
            "Capture static multi-marker observations from a local camera fleet, "
            "or import camera=image-glob pairs."
        ),
    )
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--markers", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", action="append", default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true")
    source.add_argument(
        "--images",
        action="append",
        default=[],
        metavar="CAMERA=GLOB",
        help="May be repeated; imports existing images without opening cameras.",
    )
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--notes", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_marker_manifest(args.markers)
        fleet = load_fleet(args.fleet)
        if args.live:
            dataset = _observe_live(
                fleet=fleet,
                fleet_path=args.fleet,
                manifest=manifest,
                output_path=args.out,
                camera_names=args.camera or None,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                columns=args.columns or None,
                notes=args.notes,
            )
            if dataset is None:
                return 1
        else:
            if args.camera:
                raise ValueError("--camera is only used with --live")
            dataset = _observe_images(
                fleet=fleet,
                fleet_path=args.fleet,
                manifest=manifest,
                image_specs=args.images,
                output_path=args.out,
                notes=args.notes,
            )
        save_observation_dataset(args.out, dataset)
        print(f"camcal-fleet-observe: wrote {args.out}")
        print(_graph_description(dataset))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _observe_images(
    *,
    fleet,
    fleet_path: Path,
    manifest,
    image_specs: list[str],
    output_path: Path,
    notes: str,
) -> ObservationDataset:
    parsed = [_parse_image_spec(spec) for spec in image_specs]
    names = sorted({name for name, _ in parsed})
    fleet_cameras = _select_fleet_cameras(fleet, names)
    snapshots = tuple(
        camera_snapshot_from_fleet(camera, fleet_directory=fleet_path.parent)
        for camera in fleet_cameras
    )
    snapshots_by_name = {camera.name: camera for camera in snapshots}
    detector = ArucoMultiMarkerDetector(manifest)
    observations: list[MarkerObservation] = []
    overlay_directory = output_path.parent / f"{output_path.stem}_overlays"
    overlay_directory.mkdir(parents=True, exist_ok=True)
    capture_index = 0
    for camera_name, pattern in parsed:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise ValueError(f"{camera_name} image glob matched no files: {pattern!r}")
        snapshot = snapshots_by_name[camera_name]
        for image_path_raw in paths:
            image_path = Path(image_path_raw)
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"cannot read image {image_path}")
            image_size = (int(frame.shape[1]), int(frame.shape[0]))
            if image_size != snapshot.image_size:
                raise ValueError(
                    f"{camera_name} image {image_path} is {image_size}, "
                    f"intrinsics expect {snapshot.image_size}"
                )
            detections = detector.detect(frame)
            capture_id = f"{camera_name}-{capture_index:06d}"
            captured_at = datetime.fromtimestamp(
                image_path.stat().st_mtime, timezone.utc
            ).isoformat(timespec="milliseconds")
            observations.extend(
                _make_observations(
                    camera_name=camera_name,
                    capture_id=capture_id,
                    detections=detections,
                    source_image=str(image_path.resolve()),
                    captured_at=captured_at,
                )
            )
            shown = draw_marker_detections(frame, detections, camera_name)
            cv2.imwrite(
                str(overlay_directory / f"{capture_id}.png"),
                shown,
            )
            capture_index += 1
    return ObservationDataset(
        dictionary=manifest.dictionary,
        cameras=snapshots,
        observations=tuple(observations),
        created_at=_utc_now_iso(),
        notes=notes,
    )


def _observe_live(
    *,
    fleet,
    fleet_path: Path,
    manifest,
    output_path: Path,
    camera_names: list[str] | None,
    tile_width: int,
    tile_height: int,
    columns: int | None,
    notes: str,
) -> ObservationDataset | None:
    fleet_cameras = _select_fleet_cameras(fleet, camera_names)
    snapshots = tuple(
        camera_snapshot_from_fleet(camera, fleet_directory=fleet_path.parent)
        for camera in fleet_cameras
    )
    detector = ArucoMultiMarkerDetector(manifest)
    cameras: dict[str, Camera | RealSenseCamera] = {}
    observations: list[MarkerObservation] = []
    capture_index = 0
    image_directory = output_path.parent / f"{output_path.stem}_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    try:
        connection_pairs = list(zip(fleet_cameras, snapshots, strict=True))
        connection_pairs.sort(
            key=lambda pair: pair[0].device.get("backend") != "realsense"
        )
        for fleet_camera, snapshot in connection_pairs:
            fps = _camera_fps(snapshot.profile)
            camera = _capture_camera(fleet_camera, snapshot.image_size, fps)
            camera.connect()
            if camera.image_size() != snapshot.image_size:
                raise ValueError(
                    f"camera {snapshot.name!r} streams at {camera.image_size()}, "
                    f"intrinsics expect {snapshot.image_size}"
                )
            cameras[snapshot.name] = camera

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        while True:
            current: dict[str, tuple[np.ndarray, tuple[ArucoMarkerDetection, ...]]] = {}
            tiles: list[np.ndarray] = []
            for snapshot in snapshots:
                frame = cameras[snapshot.name].capture()
                detections = detector.detect(frame)
                current[snapshot.name] = frame, detections
                tiles.append(
                    fit_frame_to_tile(
                        draw_marker_detections(frame, detections, snapshot.name),
                        tile_width=tile_width,
                        tile_height=tile_height,
                    )
                )
            live_observations = list(observations)
            for camera_name, (_, detections) in current.items():
                live_observations.extend(
                    _make_observations(
                        camera_name=camera_name,
                        capture_id="live-preview",
                        detections=detections,
                    )
                )
            preview_dataset = ObservationDataset(
                dictionary=manifest.dictionary,
                cameras=snapshots,
                observations=tuple(live_observations),
            )
            canvas = compose_preview_grid(tiles, columns=columns)
            _draw_canvas_status(
                canvas,
                f"{len(observations)} saved marker observations | "
                f"{_graph_description(preview_dataset)} | "
                "SPACE capture, ENTER save, ESC/q abort",
            )
            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return None
            if key == 13:
                if not observations:
                    continue
                return ObservationDataset(
                    dictionary=manifest.dictionary,
                    cameras=snapshots,
                    observations=tuple(observations),
                    created_at=_utc_now_iso(),
                    notes=notes,
                )
            if key != ord(' '):
                continue
            captured_at = _utc_now_iso(milliseconds=True)
            for camera_name, (frame, detections) in current.items():
                capture_id = f"{camera_name}-{capture_index:06d}"
                image_path = image_directory / f"{capture_id}.png"
                cv2.imwrite(str(image_path), frame)
                observations.extend(
                    _make_observations(
                        camera_name=camera_name,
                        capture_id=capture_id,
                        detections=detections,
                        source_image=str(image_path.relative_to(output_path.parent)),
                        captured_at=captured_at,
                    )
                )
                capture_index += 1
    finally:
        cv2.destroyAllWindows()
        for camera in cameras.values():
            camera.disconnect()


def draw_marker_detections(
    frame: np.ndarray,
    detections: tuple[ArucoMarkerDetection, ...],
    camera_name: str,
) -> np.ndarray:
    """Overlay IDs, ordered corners, and mean marker edge length."""
    shown = frame.copy()
    for detection in detections:
        corners = np.rint(detection.image_points).astype(np.int32)
        cv2.polylines(shown, [corners], True, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.circle(shown, tuple(corners[0]), 7, (0, 0, 255), -1)
        edges = np.roll(detection.image_points, -1, axis=0) - detection.image_points
        edge_px = float(np.mean(np.linalg.norm(edges, axis=1)))
        origin = tuple(corners[0] + np.array([8, -8]))
        cv2.putText(
            shown,
            f"ID {detection.marker_id}  {edge_px:.0f}px",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        shown,
        f"{camera_name}: {len(detections)} marker(s)",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shown


def _select_fleet_cameras(fleet, names: list[str] | None) -> list[FleetCamera]:
    selected = set(names) if names is not None else None
    cameras = [
        camera
        for camera in fleet.cameras
        if camera.enabled and (selected is None or camera.name in selected)
    ]
    if selected is not None:
        missing = sorted(selected - {camera.name for camera in cameras})
        if missing:
            raise ValueError(f"fleet has no enabled cameras named {missing}")
    if not cameras:
        raise ValueError("no enabled fleet cameras selected")
    return cameras


def _camera_device(camera: FleetCamera) -> int | str:
    for key in ("by_id_path", "by_path", "dev_path", "raw_device"):
        if camera.device.get(key):
            return str(camera.device[key])
    if "cv2_index" in camera.device:
        return int(camera.device["cv2_index"])
    raise ValueError(f"fleet camera {camera.name!r} has no local capture device")


def _capture_camera(
    camera: FleetCamera,
    image_size: tuple[int, int],
    fps: int,
) -> Camera | RealSenseCamera:
    backend = str(camera.device.get("backend", "opencv")).strip().lower()
    if backend == "realsense":
        serial = str(camera.device.get("serial", "")).strip()
        if not serial:
            raise ValueError(f"fleet camera {camera.name!r} has no RealSense serial")
        return RealSenseCamera(
            serial=serial,
            width=image_size[0],
            height=image_size[1],
            fps=fps,
        )
    if backend != "opencv":
        raise ValueError(
            f"fleet camera {camera.name!r} has unsupported backend {backend!r}"
        )
    return Camera(
        device=_camera_device(camera),
        width=image_size[0],
        height=image_size[1],
        fps=fps,
        fourcc=_camera_fourcc(camera.intrinsics.profile if camera.intrinsics else None),
    )


def _camera_fps(profile: dict | None) -> int:
    return int((profile or {}).get("fps", 30))


def _camera_fourcc(profile: dict | None) -> str | None:
    """Return the capture codec declared for one camera profile."""
    value = (profile or {}).get("fourcc", "MJPG")
    if value is None:
        return None
    fourcc = str(value).strip().upper()
    if fourcc in {"", "NONE"}:
        return None
    if len(fourcc) != 4:
        raise ValueError(f"camera profile fourcc must have four characters, got {value!r}")
    return fourcc


def _make_observations(
    *,
    camera_name: str,
    capture_id: str,
    detections: tuple[ArucoMarkerDetection, ...],
    source_image: str = "",
    captured_at: str = "",
) -> list[MarkerObservation]:
    return [
        MarkerObservation(
            camera_name=camera_name,
            capture_id=capture_id,
            marker_id=detection.marker_id,
            corners_px=detection.image_points,
            source_image=source_image,
            captured_at=captured_at,
            source_host=socket.gethostname(),
        )
        for detection in detections
    ]


def _parse_image_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--images expects CAMERA=GLOB, got {spec!r}")
    camera, pattern = spec.split("=", 1)
    if not camera.strip() or not pattern.strip():
        raise ValueError(f"--images expects CAMERA=GLOB, got {spec!r}")
    return camera.strip(), pattern.strip()


def _graph_description(dataset: ObservationDataset) -> str:
    graph = build_observation_graph(dataset)
    if graph.connected:
        component = graph.components[0]
        return (
            f"graph connected ({len(component.cameras)} cameras, "
            f"{len(component.markers)} markers, {len(graph.edge_counts)} edges)"
        )
    parts = "; ".join(component.describe() for component in graph.components)
    return f"graph disconnected: {parts}"


def _draw_canvas_status(image: np.ndarray, text: str) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, image.shape[0] - 42), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(
        image,
        text,
        (12, image.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _utc_now_iso(*, milliseconds: bool = False) -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds" if milliseconds else "seconds"
    )


if __name__ == "__main__":
    raise SystemExit(main())
