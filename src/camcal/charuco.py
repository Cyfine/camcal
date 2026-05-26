"""ChArUco board generation and detection.

A ChArUco board's *physical dimensions* — ``square_size_m`` and
``marker_size_m`` — are the calibration's metric ground truth. Printer
scaling, paper expansion with humidity, and ink spread move those
dimensions by 1–2%, which translates directly to commensurate
calibration error. **Always caliper-measure the printed board** and pass
those measurements into :class:`CharucoConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV ArUco dictionary names → enum values. Subset commonly used for
# calibration boards; expand on demand.
_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


@dataclass
class CharucoConfig:
    """ChArUco board specification with MEASURED physical dimensions.

    Attributes
    ----------
    squares_x, squares_y
        Number of squares along the long and short axis.
    square_size_m
        Side length of one square as measured on the printed board, in
        meters. Use the caliper reading, which captures any printer
        scaling and paper effects.
    marker_size_m
        Side length of one ArUco marker as measured on the printed board,
        in meters. Markers must be strictly smaller than squares.
    dictionary
        OpenCV ArUco dictionary name (e.g., ``DICT_5X5_100``). Must match
        the dictionary used to render the printed board.
    """

    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float
    dictionary: str = "DICT_5X5_100"

    def __post_init__(self) -> None:
        if self.squares_x <= 0 or self.squares_y <= 0:
            raise ValueError(
                f"squares_x and squares_y must be positive, got "
                f"({self.squares_x}, {self.squares_y})"
            )
        if self.square_size_m <= 0:
            raise ValueError(
                f"square_size_m must be positive, got {self.square_size_m}"
            )
        if self.marker_size_m <= 0:
            raise ValueError(
                f"marker_size_m must be positive, got {self.marker_size_m}"
            )
        if self.marker_size_m >= self.square_size_m:
            raise ValueError(
                f"marker_size_m ({self.marker_size_m}) must be smaller than "
                f"square_size_m ({self.square_size_m}); markers must fit "
                "inside squares with a white border"
            )
        if self.dictionary not in _DICTS:
            raise ValueError(
                f"Unknown ArUco dictionary '{self.dictionary}'. "
                f"Available: {sorted(_DICTS)}"
            )


def measured_dim_to_size_m(*, squares: int, measured_m: float) -> float:
    """Convert a caliper measurement of N squares into per-square size (meters).

    Example: caliper across 5 squares = 148.5 mm → ``measured_dim_to_size_m(
    squares=5, measured_m=0.1485) == 0.0297``.

    Use as many squares as practical — measurement noise distributes over
    the count.
    """
    if squares <= 0:
        raise ValueError(f"squares must be positive, got {squares}")
    if measured_m <= 0:
        raise ValueError(f"measured_m must be positive, got {measured_m}")
    return float(measured_m) / float(squares)


@dataclass
class CharucoDetection:
    """A successful detection.

    ``object_points``: (N, 3) corner coordinates in board frame (meters).
    ``image_points``: (N, 2) corner pixel coordinates.
    ``corner_ids``: (N,) the ChArUco corner indices.
    """

    object_points: np.ndarray
    image_points: np.ndarray
    corner_ids: np.ndarray


class CharucoBoard:
    """A ChArUco board built from operator-measured dimensions.

    Construct via :meth:`from_config` so the config is validated first.
    """

    def __init__(
        self,
        config: CharucoConfig,
        cv_board: cv2.aruco.CharucoBoard,
        cv_detector: cv2.aruco.CharucoDetector,
    ) -> None:
        self.config = config
        self.cv_board = cv_board
        self.cv_detector = cv_detector

    @classmethod
    def from_config(cls, config: CharucoConfig) -> CharucoBoard:
        aruco_dict = cv2.aruco.getPredefinedDictionary(_DICTS[config.dictionary])
        cv_board = cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            config.square_size_m,
            config.marker_size_m,
            aruco_dict,
        )
        cv_detector = cv2.aruco.CharucoDetector(cv_board)
        return cls(config, cv_board, cv_detector)

    # ----- generation -----

    def draw(self, pixels_per_meter: float) -> np.ndarray:
        """Render a printable board image (8-bit grayscale).

        Returned image shape:
        ``(squares_y * square_size_m * pixels_per_meter,
        squares_x * square_size_m * pixels_per_meter)``.

        Print at 100% scale and verify with calipers (see module docstring).
        """
        if pixels_per_meter <= 0:
            raise ValueError(
                f"pixels_per_meter must be positive, got {pixels_per_meter}"
            )
        width_px = round(
            self.config.squares_x * self.config.square_size_m * pixels_per_meter
        )
        height_px = round(
            self.config.squares_y * self.config.square_size_m * pixels_per_meter
        )
        return self.cv_board.generateImage((width_px, height_px))

    # ----- detection -----

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect the board; return ``(object_points, image_points)`` or None.

        Suitable inputs for ``cv2.solvePnP`` and ``cv2.calibrateCamera``.
        Returns None if fewer than 6 corners are reliably detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_corners, charuco_ids, _, _ = self.cv_detector.detectBoard(gray)
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < 6
        ):
            return None
        all_corners = self.cv_board.getChessboardCorners()  # (M, 3) in board frame
        ids = np.asarray(charuco_ids).flatten()
        object_points = all_corners[ids]
        image_points = charuco_corners.reshape(-1, 2)
        if object_points.shape[0] != image_points.shape[0]:
            return None
        return object_points.astype(np.float64), image_points.astype(np.float64)

    def detect_full(self, image: np.ndarray) -> CharucoDetection | None:
        """Same as :meth:`detect` but returns a :class:`CharucoDetection`."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_corners, charuco_ids, _, _ = self.cv_detector.detectBoard(gray)
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < 6
        ):
            return None
        all_corners = self.cv_board.getChessboardCorners()
        ids = np.asarray(charuco_ids).flatten()
        return CharucoDetection(
            object_points=all_corners[ids].astype(np.float64),
            image_points=charuco_corners.reshape(-1, 2).astype(np.float64),
            corner_ids=ids,
        )


__all__ = [
    "CharucoBoard",
    "CharucoConfig",
    "CharucoDetection",
    "measured_dim_to_size_m",
]
