import numpy as np
import pytest

from camcal import se3
from camcal.charuco import CharucoConfig
from camcal.io_yaml import (
    ExtrinsicsRecord,
    IntrinsicsRecord,
    load_board,
    load_extrinsics,
    load_intrinsics,
    save_extrinsics,
    save_intrinsics,
)


@pytest.fixture()
def board_config() -> CharucoConfig:
    return CharucoConfig(
        squares_x=5, squares_y=7,
        square_size_m=0.030, marker_size_m=0.022,
        dictionary="DICT_5X5_100",
    )


def test_intrinsics_round_trip(tmp_path):
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    d = np.array([0.01, -0.02, 0.001, -0.001, 0.0])
    rec = IntrinsicsRecord(
        K=K, d=d, image_size=(640, 480),
        reprojection_error_px=0.42, n_views=20,
        notes="round-trip test",
    )
    path = tmp_path / "intrinsics.yaml"
    save_intrinsics(path, rec)
    back = load_intrinsics(path)
    np.testing.assert_allclose(back.K, K)
    np.testing.assert_allclose(back.d, d)
    assert back.image_size == (640, 480)
    assert back.n_views == 20
    assert back.reprojection_error_px == pytest.approx(0.42)
    assert back.source == "zhang"
    assert back.notes == "round-trip test"
    assert back.created_at  # auto-filled


def test_extrinsics_round_trip(tmp_path, board_config):
    rng = np.random.default_rng(0)
    rotvec = rng.uniform(-1, 1, 3) * 0.4
    from scipy.spatial.transform import Rotation as R_scipy
    R = R_scipy.from_rotvec(rotvec).as_matrix()
    t = rng.uniform(-0.3, 0.3, 3)
    W_T_C = se3.from_Rt(R, t)

    per_frame = [se3.from_Rt(R, t + rng.normal(0, 1e-4, 3)) for _ in range(3)]

    rec = ExtrinsicsRecord(
        W_T_C=W_T_C,
        board=board_config,
        n_frames=3,
        reprojection_error_px_mean=0.31,
        reprojection_error_px_max=0.55,
        intrinsics_path="intrinsics.yaml",
        notes="extr round-trip",
        per_frame_W_T_C=per_frame,
    )
    path = tmp_path / "extrinsics.yaml"
    save_extrinsics(path, rec)
    back = load_extrinsics(path)
    np.testing.assert_allclose(back.W_T_C, W_T_C, atol=1e-12)
    assert back.board.squares_x == board_config.squares_x
    assert back.board.dictionary == board_config.dictionary
    assert back.n_frames == 3
    assert back.intrinsics_path == "intrinsics.yaml"
    assert back.notes == "extr round-trip"
    assert len(back.per_frame_W_T_C) == 3
    for orig, recovered in zip(per_frame, back.per_frame_W_T_C):
        np.testing.assert_allclose(recovered, orig, atol=1e-12)


def test_load_board(tmp_path):
    path = tmp_path / "board.yaml"
    path.write_text(
        "squares_x: 5\nsquares_y: 7\n"
        "square_size_m: 0.030\nmarker_size_m: 0.022\n"
        "dictionary: DICT_5X5_100\n"
    )
    cfg = load_board(path)
    assert cfg.squares_x == 5
    assert cfg.squares_y == 7
    assert cfg.square_size_m == pytest.approx(0.030)
    assert cfg.dictionary == "DICT_5X5_100"
