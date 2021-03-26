import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from src.model_code.factor_estimator import FactorEstimator


@pytest.fixture
def normal_input():
    Residual = np.array(
        [
            [
                -1.4367912,
                -2.6013825,
                1.33800801,
                -3.6127662,
                3.9609706,
                2.6254025,
                -0.8347457,
            ],
            [
                1.3466067,
                0.3664492,
                2.62118239,
                -1.6290464,
                -0.4641017,
                -3.6536628,
                -0.6097836,
            ],
            [
                4.500832,
                1.6229942,
                3.27592286,
                -3.6929039,
                0.2511543,
                -2.5438097,
                -0.319161,
            ],
            [
                1.8816907,
                0.6570691,
                -0.62210397,
                -1.9933962,
                -2.2156892,
                -0.6194894,
                0.2590962,
            ],
            [
                0.4839134,
                2.946613,
                2.44325189,
                0.0116231,
                -3.9395067,
                2.3658713,
                -1.7362705,
            ],
            [
                -1.9647791,
                1.2826473,
                0.05054174,
                -2.5470797,
                0.6992353,
                1.3201245,
                0.7544628,
            ],
        ]
    )
    return Residual


def test_r_hat(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    actual = factor_estimator.r_hat(rmax=6, panelty="PC", id=3)
    expect = 6
    np.testing, assert_almost_equal(actual, expect)


def test_calculate_ic(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    ic = factor_estimator._calculate_ic(r=2, id=2)
    expect = 1.34978
    np.testing, assert_almost_equal(ic, expect, decimal=6)


def test_calculate_pc(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    pc = factor_estimator._calculate_pc(r=2, rmax=5, id=2)
    expect = 1.351501
    np.testing, assert_almost_equal(pc, expect, decimal=6)


def test_calculate_f_tilde(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    f_hat = factor_estimator._calculate_f_tilde(r=2)
    expect = np.array(
        [
            [0.5681053, -1.9665053],
            [-1.1560569, -0.3777842],
            [-1.8189881, -0.8677836],
            [-0.7013869, 0.1826577],
            [-0.7135161, 0.9389870],
            [0.1761313, -0.5674743],
        ]
    )
    np.testing, assert_almost_equal(f_hat, expect)


def test_calculate_lambda_tilde(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    f_hat = np.array(
        [
            [0.5681053, 1.9665053],
            [-1.1560569, 0.3777842],
            [-1.8189881, 0.8677836],
            [-0.7013869, -0.1826577],
            [-0.7135161, -0.9389870],
            [0.1761313, 0.5674743],
        ]
    )
    lambda_hat = factor_estimator._calculate_lambda_tilde(f_hat, r=2)
    expect = np.array(
        [
            [-2.0951825, -0.05400627],
            [-1.1985173, -0.95462779],
            [-1.5878389, 0.71872749],
            [1.2482359, -2.00280008],
            [1.1363406, 2.05542389],
            [1.5535728, 0.03597738],
            [0.3335476, -0.02295224],
        ]
    )
    np.testing, assert_almost_equal(lambda_hat, expect)


def test_calculate_vkf(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    f_hat = np.array(
        [
            [0.5681053, 1.9665053],
            [-1.1560569, 0.3777842],
            [-1.8189881, 0.8677836],
            [-0.7013869, -0.1826577],
            [-0.7135161, -0.9389870],
            [0.1761313, 0.5674743],
        ]
    )
    lambda_hat = np.array(
        [
            [-2.0951825, -0.05400627],
            [-1.1985173, -0.95462779],
            [-1.5878389, 0.71872749],
            [1.2482359, -2.00280008],
            [1.1363406, 2.05542389],
            [1.5535728, 0.03597738],
            [0.3335476, -0.02295224],
        ]
    )
    vkf = factor_estimator._calculate_vkf(lambda_hat, f_hat)
    expect = 1.272006
    np.testing, assert_almost_equal(vkf, expect, decimal=6)


def test_calculate_g(normal_input):
    factor_estimator = FactorEstimator(normal_input)
    id1 = factor_estimator._calculate_g(1)
    np.testing, assert_almost_equal(id1, 0.3629848)
    id2 = factor_estimator._calculate_g(2)
    np.testing, assert_almost_equal(id2, 0.5545922)
    id3 = factor_estimator._calculate_g(3)
    np.testing, assert_almost_equal(id3, 0.2986266)
    id4 = factor_estimator._calculate_g(4)
    np.testing, assert_almost_equal(id4, 0.3333333)
