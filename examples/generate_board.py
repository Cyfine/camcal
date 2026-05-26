"""Render a printable PNG of a ChArUco board declared in board.yaml.

Usage::

    python generate_board.py board.yaml --out board.png --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from camcal.charuco import CharucoBoard
from camcal.io_yaml import load_board

INCH_PER_METER = 39.3701


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("board_yaml", type=Path, help="Path to board.yaml.")
    p.add_argument(
        "--out", type=Path, default=Path("board.png"),
        help="Output PNG path (default: board.png).",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Print resolution in dots-per-inch (default: 300).",
    )
    args = p.parse_args(argv)

    config = load_board(args.board_yaml)
    board = CharucoBoard.from_config(config)
    pixels_per_meter = args.dpi * INCH_PER_METER
    img = board.draw(pixels_per_meter=pixels_per_meter)
    cv2.imwrite(str(args.out), img)
    print(
        f"wrote {args.out}  "
        f"({img.shape[1]} x {img.shape[0]} px at {args.dpi} dpi  "
        f"= {config.squares_x * config.square_size_m * 1000:.1f} x "
        f"{config.squares_y * config.square_size_m * 1000:.1f} mm)"
    )
    print(
        "PRINT AT 100% SCALE (uncheck 'fit to page' / 'scale to fit'), "
        "then caliper-measure and update board.yaml."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
