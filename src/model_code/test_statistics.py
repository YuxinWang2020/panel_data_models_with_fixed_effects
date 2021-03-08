import numpy as np

from src.model_code.statistics import caculate_rmse
from src.model_code.statistics import caculate_sde


def test_caculate_rmse_1d():
    estimated = [1, 2, 3, 4, 5]
    real = [1, 3, 5, 2, 4]
    expected = np.array([0, 1, 2, 2, 1])
    rmse = caculate_rmse(estimated, real)
    np.testing.assert_array_almost_equal(rmse, expected)


def test_caculate_rmse_2d():
    estimated = [[1, 2, 3, 4, 5], [3, 5, 6, 2, 3]]
    real = [1, 3, 5, 2, 4]
    expected = np.array([1.414214, 1.581139, 1.581139, 1.414214, 1.000000])
    rmse = caculate_rmse(estimated, real)
    np.testing.assert_array_almost_equal(rmse, expected)


def test_caculate_sde_2d():
    estimated = [[1, 2, 3, 4, 5], [3, 5, 6, 2, 3]]
    expected = np.array([0.70710678, 1.06066017, 1.06066017, 0.70710678, 0.70710678])
    sde = caculate_sde(estimated)
    np.testing.assert_array_almost_equal(sde, expected)
