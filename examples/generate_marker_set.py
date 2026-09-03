"""Create an editable marker manifest and print-exact A3 PNG sheets.

Usage::

    python examples/generate_marker_set.py marker_prints

If ``marker_prints/markers.yaml`` already exists, its IDs and nominal sizes
are used.  Otherwise the command creates the default DICT_6X6_250 IDs 0--9
manifest before rendering.  Print at 100% scale, measure each black square,
and enter the measured values in ``markers.yaml`` before calibration.
"""

from camcal.generate_markers import main


if __name__ == "__main__":
    raise SystemExit(main())
