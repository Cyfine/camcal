"""Print-exact A3 PNG generation for standalone workspace markers."""

from __future__ import annotations

from pathlib import Path
import struct
import zlib

import cv2
import numpy as np

from .charuco import aruco_dictionary
from .marker_map import MarkerManifest, MarkerSpec


A3_PORTRAIT_MM = (297.0, 420.0)
MM_PER_INCH = 25.4


def render_a3_marker_sheet(
    marker: MarkerSpec,
    *,
    dictionary: str,
    dpi: int = 300,
) -> np.ndarray:
    """Render one portrait A3 sheet as an 8-bit grayscale image.

    The ArUco black square has exactly ``marker.nominal_size_m`` at the
    declared DPI.  Human-readable text and the orientation arrow are outside
    a 20 mm white region below the marker.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    cv_dictionary = aruco_dictionary(dictionary)
    capacity = int(cv_dictionary.bytesList.shape[0])
    if marker.marker_id >= capacity:
        raise ValueError(
            f"marker ID {marker.marker_id} exceeds {dictionary} range 0..{capacity - 1}"
        )

    paper_w_px = _mm_to_px(A3_PORTRAIT_MM[0], dpi)
    paper_h_px = _mm_to_px(A3_PORTRAIT_MM[1], dpi)
    marker_px = _mm_to_px(marker.nominal_size_m * 1000.0, dpi)
    x0 = (paper_w_px - marker_px) // 2
    y0 = _mm_to_px(35.0, dpi)
    if x0 < _mm_to_px(10.0, dpi) or y0 + marker_px > _mm_to_px(300.0, dpi):
        raise ValueError(
            f"{marker.nominal_size_m:.3f} m marker does not fit the A3 layout"
        )

    canvas = np.full((paper_h_px, paper_w_px), 255, dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(
        cv_dictionary,
        marker.marker_id,
        marker_px,
    )
    canvas[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker_image

    label_y = y0 + marker_px + _mm_to_px(32.0, dpi)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = f"ARUCO ID {marker.marker_id:03d}"
    detail = (
        f"{dictionary}   nominal black square "
        f"{marker.nominal_size_m * 1000.0:.1f} mm"
    )
    _put_centered_text(canvas, title, label_y, font, 1.2 * dpi / 300.0, dpi)
    _put_centered_text(
        canvas,
        detail,
        label_y + _mm_to_px(13.0, dpi),
        font,
        0.52 * dpi / 300.0,
        dpi,
    )

    arrow_x = paper_w_px // 2
    arrow_bottom = label_y + _mm_to_px(54.0, dpi)
    arrow_top = arrow_bottom - _mm_to_px(18.0, dpi)
    cv2.arrowedLine(
        canvas,
        (arrow_x, arrow_bottom),
        (arrow_x, arrow_top),
        0,
        max(2, round(dpi / 100)),
        tipLength=0.35,
    )
    _put_centered_text(
        canvas,
        "TOP / +Y",
        arrow_bottom + _mm_to_px(10.0, dpi),
        font,
        0.60 * dpi / 300.0,
        dpi,
    )
    return canvas


def write_print_exact_png(path: str | Path, image: np.ndarray, *, dpi: int) -> None:
    """Write a PNG with a physical-resolution ``pHYs`` chunk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode {path}")
    png = encoded.tobytes()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("OpenCV returned invalid PNG data")
    pixels_per_metre = round(float(dpi) / 0.0254)
    data = struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1)
    kind = b"pHYs"
    chunk = (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )
    ihdr_length = struct.unpack(">I", png[8:12])[0]
    after_ihdr = 8 + 12 + ihdr_length
    path.write_bytes(png[:after_ihdr] + chunk + png[after_ihdr:])


def generate_a3_marker_set(
    output_directory: str | Path,
    manifest: MarkerManifest,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Write one A3 PNG for every marker in ``manifest``."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for marker in manifest.markers:
        path = output_directory / f"aruco_{marker.marker_id:03d}_a3.png"
        image = render_a3_marker_sheet(
            marker,
            dictionary=manifest.dictionary,
            dpi=dpi,
        )
        write_print_exact_png(path, image, dpi=dpi)
        paths.append(path)
    return paths


def _mm_to_px(value_mm: float, dpi: int) -> int:
    return round(float(value_mm) * float(dpi) / MM_PER_INCH)


def _put_centered_text(
    image: np.ndarray,
    text: str,
    baseline_y: int,
    font: int,
    scale: float,
    dpi: int,
) -> None:
    thickness = max(1, round(dpi / 150))
    (width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(
        image,
        text,
        ((image.shape[1] - width) // 2, baseline_y),
        font,
        scale,
        0,
        thickness,
        cv2.LINE_AA,
    )


__all__ = [
    "A3_PORTRAIT_MM",
    "generate_a3_marker_set",
    "render_a3_marker_sheet",
    "write_print_exact_png",
]
