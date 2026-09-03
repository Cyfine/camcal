import numpy as np

from camcal import se3
from camcal.calibrate_stereo_extrinsics import relative_camera_transform


def test_relative_camera_transform_cancels_shared_target_frame():
    C0_T_target = np.eye(4)
    C0_T_target[:3, 3] = [0.1, -0.2, 1.2]

    C1_T_C0 = np.eye(4)
    C1_T_C0[:3, 3] = [0.5, 0.0, 0.1]
    C1_T_target = C1_T_C0 @ C0_T_target

    actual = relative_camera_transform(C0_T_target, C1_T_target)

    se3.assert_se3(actual)
    np.testing.assert_allclose(actual, C1_T_C0, atol=1e-12)
