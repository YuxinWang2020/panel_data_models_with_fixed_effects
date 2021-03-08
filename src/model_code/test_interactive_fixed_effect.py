import numpy as np
import pytest

from src.model_code.interactive_fixed_effect import InteractiveFixedEffect


@pytest.fixture
def normal_input():
    X = np.array(
        [
            [
                [-1.18775541, 0.13930048, -1.0111979, -2.79614049],
                [-3.31437866, -0.29828577, -3.60858373, -2.40489821],
                [-0.48556501, -3.27345056, 4.61644392, 0.70666412],
            ],
            [
                [-0.10509627, -2.04091154, -1.43984235, -2.80657453],
                [-2.14144368, -1.43833089, 0.45613916, -2.51124922],
                [-1.36104357, -1.9053894, 4.58706261, 1.85836145],
            ],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [
                [0.65982146, -2.65987201, -0.94660116, -0.78080841],
                [0.65982146, -2.65987201, -0.94660116, -0.78080841],
                [0.65982146, -2.65987201, -0.94660116, -0.78080841],
            ],
            [
                [-2.85110504, -2.85110504, -2.85110504, -2.85110504],
                [-3.34376827, -3.34376827, -3.34376827, -3.34376827],
                [1.79288486, 1.79288486, 1.79288486, 1.79288486],
            ],
        ]
    )
    Y = np.array(
        [
            [-8.63640022, -16.01744339, -18.03955115, -21.83472808],
            [-19.09376238, -19.36575265, -13.15849355, -19.41834245],
            [7.61382859, -4.62878659, 26.65879902, 15.83486644],
        ]
    )
    beta = (1, 3, 5, 2, 4)
    input = {"X": X, "Y": Y, "beta": beta}
    return input


def test_fit_beta_hat(normal_input):
    interactive_estimator = InteractiveFixedEffect(normal_input["Y"], normal_input["X"])
    beta_hat, beta_hat_list, f_hat, lambda_hat = interactive_estimator.fit(
        r=2, beta_hat_0=normal_input["beta"]
    )
    expect_beta_hat = np.array([[1.523508, 2.609971, 4.903340, 1.522600, 4.158032]])
    np.testing.assert_array_almost_equal(beta_hat, expect_beta_hat)


def test_calculate_f_hat(normal_input):
    interactive_estimator = InteractiveFixedEffect(normal_input["Y"], normal_input["X"])
    f_hat = interactive_estimator._calculate_f_hat(
        np.array([normal_input["beta"]]), r=2
    )
    expect = np.array(
        [[1.5482942, 0.7668864], [0.3597737, -0.9481272], [0.6880028, -1.2300162]]
    )
    np.testing.assert_array_almost_equal(f_hat, expect)


def test_calculate_lambda_hat(normal_input):
    interactive_estimator = InteractiveFixedEffect(normal_input["Y"], normal_input["X"])
    f_hat = np.array(
        [[1.5482942, 0.7668864], [0.3597737, -0.9481272], [0.6880028, -1.2300162]]
    )
    lambda_hat = interactive_estimator._calculate_lambda_hat(
        np.array([normal_input["beta"]]), f_hat, r=2
    )
    expect = np.array(
        [
            [-1.6331689, 0.7396594],
            [0.1741834, 1.7876252],
            [-2.8125904, -0.1033363],
            [-1.5566757, -0.3892726],
        ]
    )
    np.testing.assert_array_almost_equal(lambda_hat, expect)


def test_calculate_beta_hat(normal_input):
    interactive_estimator = InteractiveFixedEffect(normal_input["Y"], normal_input["X"])
    f_hat = np.array(
        [[1.5482942, 0.7668864], [0.3597737, -0.9481272], [0.6880028, -1.2300162]]
    )
    lambda_hat = np.array(
        [
            [-1.6331689, 0.7396594],
            [0.1741834, 1.7876252],
            [-2.8125904, -0.1033363],
            [-1.5566757, -0.3892726],
        ]
    )
    beta_hat = interactive_estimator._calculate_beta_hat(f_hat, lambda_hat)
    expect = np.array([[1.094433, 2.897426, 4.978414, 1.978000, 3.971552]])
    np.testing.assert_array_almost_equal(beta_hat, expect)
