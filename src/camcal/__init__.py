"""camcal — ChArUco-based camera calibration.

Two operator-facing entry points:

* :mod:`camcal.calibrate_intrinsics` — Zhang's method (K + distortion).
* :mod:`camcal.calibrate_extrinsics` — pose of a camera in a world frame
  defined by a ChArUco board (computes ``W_T_C``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
