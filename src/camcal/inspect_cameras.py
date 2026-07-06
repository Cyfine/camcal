"""Live tiled preview of all connected cameras.

This is an operator-facing inspection tool that complements
``camcal-list-cameras``: it opens every discovered camera and renders the
streams into one tiled viewport so the operator can verify framing, device
identity, and basic stream health before calibration.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera, CameraError
from .list_cameras import CameraInfo, list_cameras

KEY_ESC = 27
KEY_Q = ord("q")
WINDOW_TITLE = "camcal: inspect cameras"
DEFAULT_TILE_WIDTH = 480
DEFAULT_TILE_HEIGHT = 270
GRID_GAP_PX = 12
DEFAULT_TILE_OVERLAY_WIDTH_RATIO = 0.62
DEFAULT_TILE_OVERLAY_HEIGHT_RATIO = 0.15
DEFAULT_CANVAS_OVERLAY_WIDTH_RATIO = 0.44
DEFAULT_CANVAS_OVERLAY_HEIGHT_RATIO = 0.06


@dataclass
class PreviewSource:
    info: CameraInfo
    device: str
    camera: Camera | None
    status: str


@dataclass(frozen=True)
class OverlayLayout:
    lines: list[str]
    font_scale: float
    thickness: int
    pad_x: int
    pad_y: int
    line_gap: int
    box_width: int
    box_height: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-inspect-cameras",
        description=(
            "Open all discovered cameras in a tiled viewport so the operator "
            "can verify identity and framing before calibration."
        ),
    )
    p.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    p.add_argument("--height", type=int, default=720, help="Requested capture height.")
    p.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    p.add_argument(
        "--tile-width",
        type=int,
        default=DEFAULT_TILE_WIDTH,
        help=f"Tile width in pixels (default {DEFAULT_TILE_WIDTH}).",
    )
    p.add_argument(
        "--tile-height",
        type=int,
        default=DEFAULT_TILE_HEIGHT,
        help=f"Tile height in pixels (default {DEFAULT_TILE_HEIGHT}).",
    )
    p.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Optional fixed grid column count. Default is a near-square layout.",
    )
    return p


def preview_grid_shape(count: int, *, columns: int | None = None) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a tiled preview grid."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if columns is not None and columns <= 0:
        raise ValueError(f"columns must be positive when provided, got {columns}")

    cols = min(columns, count) if columns is not None else math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def fit_frame_to_tile(
    frame: np.ndarray,
    *,
    tile_width: int,
    tile_height: int,
    background_bgr: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Letterbox ``frame`` into a tile of exactly ``tile_width x tile_height``."""
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError(
            f"tile_width and tile_height must be positive, got "
            f"{tile_width}x{tile_height}"
        )

    in_h, in_w = frame.shape[:2]
    scale = min(tile_width / in_w, tile_height / in_h)
    out_w = max(1, int(round(in_w * scale)))
    out_h = max(1, int(round(in_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (out_w, out_h), interpolation=interp)

    tile = np.full((tile_height, tile_width, 3), background_bgr, dtype=np.uint8)
    off_x = (tile_width - out_w) // 2
    off_y = (tile_height - out_h) // 2
    tile[off_y:off_y + out_h, off_x:off_x + out_w] = resized
    return tile


def compose_preview_grid(
    tiles: list[np.ndarray],
    *,
    columns: int | None = None,
    gap_px: int = GRID_GAP_PX,
) -> np.ndarray:
    """Compose same-sized tiles into one preview canvas."""
    if not tiles:
        raise ValueError("compose_preview_grid requires at least one tile")
    if gap_px < 0:
        raise ValueError(f"gap_px must be non-negative, got {gap_px}")

    tile_h, tile_w = tiles[0].shape[:2]
    for tile in tiles[1:]:
        if tile.shape[:2] != (tile_h, tile_w):
            raise ValueError("all tiles must share the same size")

    rows, cols = preview_grid_shape(len(tiles), columns=columns)
    canvas_h = rows * tile_h + max(0, rows - 1) * gap_px
    canvas_w = cols * tile_w + max(0, cols - 1) * gap_px
    canvas = np.full((canvas_h, canvas_w, 3), 10, dtype=np.uint8)

    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        y0 = row * (tile_h + gap_px)
        x0 = col * (tile_w + gap_px)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_columns = args.columns or None

    cameras = list_cameras()
    if not cameras:
        if sys.platform != "linux":
            print(
                "Camera inspection currently relies on Linux V4L2 discovery.",
                file=sys.stderr,
            )
        else:
            print("No cameras found.", file=sys.stderr)
        return 2

    sources = _connect_sources(
        cameras,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    try:
        while True:
            tiles = [
                _render_tile(
                    source,
                    index=index,
                    tile_width=args.tile_width,
                    tile_height=args.tile_height,
                )
                for index, source in enumerate(sources)
            ]
            canvas = compose_preview_grid(tiles, columns=requested_columns)
            _draw_overlay_badge(
                canvas,
                [f"{len(sources)} camera(s)  ESC/q close"],
                origin="bottom-left",
                max_width_ratio=DEFAULT_CANVAS_OVERLAY_WIDTH_RATIO,
                max_height_ratio=DEFAULT_CANVAS_OVERLAY_HEIGHT_RATIO,
            )
            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (KEY_ESC, KEY_Q):
                return 0
    finally:
        cv2.destroyWindow(WINDOW_TITLE)
        for source in sources:
            if source.camera is not None:
                source.camera.disconnect()


def _connect_sources(
    cameras: list[CameraInfo],
    *,
    width: int,
    height: int,
    fps: int,
) -> list[PreviewSource]:
    sources: list[PreviewSource] = []
    for info in cameras:
        device = info.by_id_path or info.dev_path
        camera = Camera(device=device, width=width, height=height, fps=fps)
        try:
            camera.connect()
        except CameraError as exc:
            sources.append(
                PreviewSource(
                    info=info,
                    device=device,
                    camera=None,
                    status=str(exc),
                )
            )
            continue
        sources.append(
            PreviewSource(
                info=info,
                device=device,
                camera=camera,
                status="streaming",
            )
        )
    return sources


def _render_tile(
    source: PreviewSource,
    *,
    index: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    frame: np.ndarray | None = None
    if source.camera is not None:
        try:
            frame = source.camera.capture()
            source.status = "streaming"
        except CameraError as exc:
            source.status = str(exc)
            source.camera.disconnect()
            source.camera = None

    if frame is None:
        tile = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    else:
        tile = fit_frame_to_tile(
            frame,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    label = source.info.model or "Unknown camera"
    if source.info.serial:
        label = f"{label} [{source.info.serial}]"
    lines = [
        f"[{index}] {label}",
        _status_line(source, frame),
    ]
    _draw_overlay_badge(
        tile,
        lines,
        origin="top-left",
        max_width_ratio=DEFAULT_TILE_OVERLAY_WIDTH_RATIO,
        max_height_ratio=DEFAULT_TILE_OVERLAY_HEIGHT_RATIO,
    )
    return tile


def _status_line(source: PreviewSource, frame: np.ndarray | None) -> str:
    if frame is not None:
        h, w = frame.shape[:2]
        return f"{w}x{h}  {source.device}"
    return f"unavailable  {source.status}"


def _truncate(text: str, *, max_chars: int) -> str:
    if max_chars <= 3 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 3]}..."


def compute_overlay_layout(
    img_shape: tuple[int, int, int] | tuple[int, int],
    lines: list[str],
    *,
    max_width_ratio: float,
    max_height_ratio: float,
) -> OverlayLayout:
    """Fit a compact text badge into a bounded fraction of the image area."""
    if not lines:
        raise ValueError("compute_overlay_layout requires at least one line")
    if not (0.0 < max_width_ratio <= 1.0):
        raise ValueError(f"max_width_ratio must be in (0, 1], got {max_width_ratio}")
    if not (0.0 < max_height_ratio <= 1.0):
        raise ValueError(f"max_height_ratio must be in (0, 1], got {max_height_ratio}")

    img_h, img_w = img_shape[:2]
    min_dim = min(img_h, img_w)
    max_box_w = max(40, int(round(img_w * max_width_ratio)))
    max_box_h = max(24, int(round(img_h * max_height_ratio)))
    thickness = 1
    pad_x = max(6, int(round(min_dim * 0.025)))
    pad_y = max(5, int(round(min_dim * 0.022)))
    line_gap = max(2, int(round(min_dim * 0.008)))

    base_scale = float(np.clip(min_dim / 520.0, 0.34, 0.78))
    for attempt in range(18):
        font_scale = max(0.28, base_scale - 0.03 * attempt)
        max_text_w = max_box_w - 2 * pad_x
        fitted_lines = [
            truncate_to_pixel_width(
                line,
                max_width_px=max_text_w,
                font_scale=font_scale,
                thickness=thickness,
            )
            for line in lines
        ]
        text_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            for line in fitted_lines
        ]
        text_width = max(width for width, _ in text_sizes)
        text_height = sum(height for _, height in text_sizes)
        if len(text_sizes) > 1:
            text_height += line_gap * (len(text_sizes) - 1)
        box_width = text_width + 2 * pad_x
        box_height = text_height + 2 * pad_y
        if box_width <= max_box_w and box_height <= max_box_h:
            return OverlayLayout(
                lines=fitted_lines,
                font_scale=font_scale,
                thickness=thickness,
                pad_x=pad_x,
                pad_y=pad_y,
                line_gap=line_gap,
                box_width=box_width,
                box_height=box_height,
            )

    font_scale = 0.28
    max_text_w = max_box_w - 2 * pad_x
    fitted_lines = [
        truncate_to_pixel_width(
            line,
            max_width_px=max_text_w,
            font_scale=font_scale,
            thickness=thickness,
        )
        for line in lines
    ]
    text_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in fitted_lines
    ]
    text_width = max(width for width, _ in text_sizes)
    text_height = sum(height for _, height in text_sizes)
    if len(text_sizes) > 1:
        text_height += line_gap * (len(text_sizes) - 1)
    return OverlayLayout(
        lines=fitted_lines,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=pad_x,
        pad_y=pad_y,
        line_gap=line_gap,
        box_width=min(max_box_w, text_width + 2 * pad_x),
        box_height=min(max_box_h, text_height + 2 * pad_y),
    )


def truncate_to_pixel_width(
    text: str,
    *,
    max_width_px: int,
    font_scale: float,
    thickness: int,
) -> str:
    """Truncate text with an ellipsis so it fits a target pixel width."""
    if max_width_px <= 0:
        return ""
    width, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0]
    if width <= max_width_px:
        return text

    ellipsis = "..."
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = f"{text[:mid]}{ellipsis}"
        cand_w, _ = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0]
        if cand_w <= max_width_px:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_overlay_badge(
    img: np.ndarray,
    lines: list[str],
    *,
    origin: str,
    max_width_ratio: float,
    max_height_ratio: float,
) -> None:
    layout = compute_overlay_layout(
        img.shape,
        lines,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio,
    )

    margin = max(8, int(round(min(img.shape[:2]) * 0.02)))
    if origin == "top-left":
        x0 = margin
        y0 = margin
    elif origin == "bottom-left":
        x0 = margin
        y0 = img.shape[0] - margin - layout.box_height
    else:
        raise ValueError(f"Unsupported origin {origin!r}")

    x1 = min(img.shape[1], x0 + layout.box_width)
    y1 = min(img.shape[0], y0 + layout.box_height)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.54, img, 0.46, 0, img)

    cursor_y = y0 + layout.pad_y
    for line in layout.lines:
        (_, text_h), baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            layout.font_scale,
            layout.thickness,
        )
        cursor_y += text_h
        cv2.putText(
            img,
            line,
            (x0 + layout.pad_x, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            layout.font_scale,
            (255, 255, 255),
            layout.thickness,
            cv2.LINE_AA,
        )
        cursor_y += baseline + layout.line_gap


__all__ = [
    "OverlayLayout",
    "compute_overlay_layout",
    "compose_preview_grid",
    "fit_frame_to_tile",
    "main",
    "preview_grid_shape",
    "truncate_to_pixel_width",
]


if __name__ == "__main__":
    sys.exit(main())
