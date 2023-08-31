import pytest
from lh2_3d_analysis import solve_3d_scene
import numpy as np


INOUT  = [
    (
        # In
        np.array([[0.0, 0.0], [-0.3333333333333333, 0.0], [0.3333333333333333, 0.0], [0.0, 0.0], [0.5, 0.0], [0.2, 0.0], [0.0, 0.7071067811865474], [-0.3333333333333333, 0.4472135954999579], [0.3333333333333333, 0.4472135954999579], [0.0, 0.35355339059327373], [0.5, 0.3162277660168379], [0.2, 0.2773500981126145]]), # in
        np.array([[0.3333333333333334, 0.0], [7.850462293418876e-17, 0.0], [7.850462293418876e-17, 0.0], [-0.3333333333333332, 0.0], [-0.1999999999999999, 0.0], [-0.4999999999999999, 0.0], [0.3333333333333334, 0.447213595499958], [7.850462293418876e-17, 0.7071067811865476], [7.850462293418876e-17, 0.3535533905932738], [-0.3333333333333332, 0.4472135954999579], [-0.1999999999999999, 0.2773500981126145], [-0.4999999999999999, 0.3162277660168379]]),
        # Out
        np.array([1,0,0]),                       # true_t
        np.array([[0,-1,0],[1,0,0],[0,0,1]]),    # true_R
    ),

        ]

@pytest.mark.parametrize("pts_a,pts_b,true_t,true_R", INOUT)
def test_solve_3d_scene(pts_a, pts_b, true_t, true_R):

    # Run function
    _, t_star, R_star = solve_3d_scene(pts_a, pts_b)

    # Assert errors
    np.testing.assert_array_almost_equal(true_R, R_star)
    np.testing.assert_array_almost_equal(true_t, t_star)
