"""Command-line generation of print-exact A3 workspace markers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .marker_assets import generate_a3_marker_set
from .marker_map import (
    default_marker_manifest,
    load_marker_manifest,
    save_marker_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate A3 ArUco sheets for fleet-camera calibration."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Existing marker YAML; defaults to OUTPUT_DIRECTORY/markers.yaml.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    manifest_path = args.manifest or args.output_directory / "markers.yaml"
    if manifest_path.exists():
        manifest = load_marker_manifest(manifest_path)
    elif args.manifest is not None:
        parser.error(f"manifest does not exist: {manifest_path}")
    else:
        manifest = default_marker_manifest()
        save_marker_manifest(manifest_path, manifest)

    paths = generate_a3_marker_set(args.output_directory, manifest, dpi=args.dpi)
    print(f"wrote {len(paths)} A3 PNG sheets")
    print(f"marker manifest: {manifest_path}")
    print("Print at 100% scale, measure every black square, then edit the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
