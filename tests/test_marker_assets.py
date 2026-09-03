from __future__ import annotations

import struct

import cv2

from camcal.generate_markers import main
from camcal.marker_assets import (
    A3_PORTRAIT_MM,
    generate_a3_marker_set,
    render_a3_marker_sheet,
)
from camcal.marker_map import MarkerManifest, MarkerSpec


def test_rendered_sheet_has_exact_a3_pixels_and_detectable_marker():
    dpi = 100
    marker = MarkerSpec(6, nominal_size_m=0.200)
    image = render_a3_marker_sheet(marker, dictionary="DICT_6X6_250", dpi=dpi)

    assert image.shape == (
        round(A3_PORTRAIT_MM[1] * dpi / 25.4),
        round(A3_PORTRAIT_MM[0] * dpi / 25.4),
    )
    corners, ids, _ = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    ).detectMarkers(image)
    assert ids is not None
    assert ids.reshape(-1).tolist() == [6]
    assert len(corners) == 1


def test_generated_png_embeds_requested_physical_resolution(tmp_path):
    dpi = 120
    manifest = MarkerManifest(
        "DICT_6X6_250",
        (MarkerSpec(3, nominal_size_m=0.180),),
    )

    paths = generate_a3_marker_set(tmp_path, manifest, dpi=dpi)

    assert [path.name for path in paths] == ["aruco_003_a3.png"]
    data = paths[0].read_bytes()
    chunk_index = data.index(b"pHYs")
    x_ppm, y_ppm, unit = struct.unpack(">IIB", data[chunk_index + 4 : chunk_index + 13])
    assert x_ppm == round(dpi / 0.0254)
    assert y_ppm == x_ppm
    assert unit == 1
    assert cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE) is not None


def test_operator_command_uses_an_existing_manifest(tmp_path):
    manifest_path = tmp_path / "selected.yaml"
    manifest_path.write_text(
        "dictionary: DICT_6X6_250\n"
        "markers:\n"
        "- id: 23\n"
        "  nominal_black_square_side_m: 0.18\n"
        "  measured_black_square_side_m: 0.1795\n"
    )
    output_directory = tmp_path / "prints"

    result = main(
        [str(output_directory), "--manifest", str(manifest_path), "--dpi", "60"]
    )

    assert result == 0
    assert (output_directory / "aruco_023_a3.png").is_file()
    assert not (output_directory / "markers.yaml").exists()
