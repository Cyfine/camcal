import cv2
import numpy as np

from camcal.inspect_cameras import (
    compute_overlay_layout,
    compose_preview_grid,
    fit_frame_to_tile,
    preview_grid_shape,
    truncate_to_pixel_width,
)


def test_preview_grid_shape_prefers_near_square_layout():
    assert preview_grid_shape(1) == (1, 1)
    assert preview_grid_shape(2) == (1, 2)
    assert preview_grid_shape(3) == (2, 2)
    assert preview_grid_shape(5) == (2, 3)


def test_preview_grid_shape_respects_fixed_columns():
    assert preview_grid_shape(4, columns=1) == (4, 1)
    assert preview_grid_shape(4, columns=3) == (2, 3)


def test_fit_frame_to_tile_letterboxes_without_distortion():
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    tile = fit_frame_to_tile(
        frame,
        tile_width=120,
        tile_height=120,
        background_bgr=(0, 0, 0),
    )
    assert tile.shape == (120, 120, 3)
    assert np.all(tile[0, 0] == 0)
    assert np.all(tile[60, 60] == 255)


def test_compose_preview_grid_uses_gap_and_tile_dimensions():
    tiles = [
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20, 3), dtype=np.uint8),
    ]
    canvas = compose_preview_grid(tiles, columns=2, gap_px=4)
    assert canvas.shape == (24, 44, 3)


def test_compute_overlay_layout_stays_within_requested_area():
    layout = compute_overlay_layout(
        (270, 480, 3),
        ["[0] Intel RealSense D415", "1920x1080  /dev/v4l/by-id/usb-very-long"],
        max_width_ratio=0.62,
        max_height_ratio=0.15,
    )
    assert layout.box_width <= round(480 * 0.62)
    assert layout.box_height <= round(270 * 0.15)


def test_truncate_to_pixel_width_respects_pixel_budget():
    text = truncate_to_pixel_width(
        "abcdefghijklmnopqrstuvwxyz0123456789",
        max_width_px=120,
        font_scale=0.4,
        thickness=1,
    )
    width, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    assert width <= 120
