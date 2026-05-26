import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R_scipy

from camcal import se3


def _random_se3(rng: np.random.Generator) -> np.ndarray:
    rotvec = rng.uniform(-1.0, 1.0, size=3) * np.pi / 2
    R = R_scipy.from_rotvec(rotvec).as_matrix()
    t = rng.uniform(-0.5, 0.5, size=3)
    return se3.from_Rt(R, t)


def test_is_valid_se3_rejects_garbage():
    assert not se3.is_valid_se3(np.eye(3))
    assert not se3.is_valid_se3(np.full((4, 4), 0.5))
    assert se3.is_valid_se3(np.eye(4))


def test_inverse_round_trip():
    rng = np.random.default_rng(0)
    T = _random_se3(rng)
    np.testing.assert_allclose(
        se3.compose(T, se3.inverse(T)), np.eye(4), atol=1e-9,
    )
    np.testing.assert_allclose(
        se3.compose(se3.inverse(T), T), np.eye(4), atol=1e-9,
    )


def test_rvec_tvec_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = _random_se3(rng)
        rvec, tvec = se3.to_rvec_tvec(T)
        T_back = se3.from_rvec_tvec(rvec, tvec)
        np.testing.assert_allclose(T_back, T, atol=1e-9)


def test_quat_t_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        T = _random_se3(rng)
        q, t = se3.to_quat_t(T)
        T_back = se3.from_quat_t(q, t)
        np.testing.assert_allclose(T_back, T, atol=1e-9)


def test_mean_se3_recovers_centre():
    rng = np.random.default_rng(3)
    centre = _random_se3(rng)
    Ts = []
    for _ in range(8):
        # Small symmetric perturbation around centre — Karcher mean recovers it.
        rot = R_scipy.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix()
        t = rng.normal(0, 0.005, 3)
        delta = se3.from_Rt(rot, t)
        Ts.append(se3.compose(centre, delta))
        Ts.append(se3.compose(centre, se3.inverse(delta)))
    mean = se3.mean_se3(Ts)
    assert se3.translation_distance(mean, centre) < 1e-3
    assert se3.rotation_distance(mean, centre) < 0.1


def test_project_to_se3_snaps_to_orthonormal():
    R_noisy = np.eye(3) + 1e-5 * np.random.default_rng(4).standard_normal((3, 3))
    T = np.eye(4)
    T[:3, :3] = R_noisy
    T[:3, 3] = [0.1, 0.2, 0.3]
    T_clean = se3.project_to_se3(T)
    assert se3.is_valid_se3(T_clean)
    # Translation is preserved.
    np.testing.assert_allclose(T_clean[:3, 3], [0.1, 0.2, 0.3])


def test_project_to_se3_refuses_garbage():
    T = np.eye(4)
    T[:3, :3] = 2 * np.eye(3)  # not close to orthonormal
    with pytest.raises(ValueError):
        se3.project_to_se3(T)
