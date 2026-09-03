from __future__ import annotations

import pytest

from camcal.marker_map import (
    MarkerManifest,
    MarkerSpec,
    default_marker_manifest,
    load_marker_manifest,
    save_marker_manifest,
)


def test_default_manifest_has_unique_6x6_ids_zero_through_nine():
    manifest = default_marker_manifest()

    assert manifest.dictionary == "DICT_6X6_250"
    assert [marker.marker_id for marker in manifest.markers] == list(range(10))
    assert len(manifest.by_id) == 10


def test_manifest_round_trip_preserves_editable_measured_size(tmp_path):
    manifest = MarkerManifest(
        dictionary="DICT_6X6_250",
        markers=(
            MarkerSpec(4, nominal_size_m=0.240, measured_size_m=0.2387),
            MarkerSpec(19, nominal_size_m=0.220),
        ),
    )
    path = tmp_path / "markers.yaml"

    save_marker_manifest(path, manifest)
    loaded = load_marker_manifest(path)

    assert loaded == manifest
    assert loaded.get(4).side_length_m == pytest.approx(0.2387)
    assert loaded.get(4).uses_measured_size
    assert loaded.get(19).side_length_m == pytest.approx(0.220)
    assert not loaded.get(19).uses_measured_size
    assert "measured_black_square_side_m" in path.read_text()


def test_manifest_rejects_duplicate_and_out_of_dictionary_ids():
    with pytest.raises(ValueError, match="unique"):
        MarkerManifest(
            "DICT_6X6_250",
            (MarkerSpec(3, 0.2), MarkerSpec(3, 0.2)),
        )
    with pytest.raises(ValueError, match="range"):
        MarkerManifest("DICT_6X6_250", (MarkerSpec(250, 0.2),))


def test_manifest_get_reports_unconfigured_id():
    manifest = default_marker_manifest(marker_ids=(7, 42))

    with pytest.raises(KeyError, match="41"):
        manifest.get(41)
