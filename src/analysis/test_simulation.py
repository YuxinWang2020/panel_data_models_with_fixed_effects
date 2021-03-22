import numpy as np

from src.analysis.monte_carlo_dgp import dgp_additive_fixed_effects_model
from src.analysis.monte_carlo_dgp import dgp_interactive_fixed_effects_model
from src.analysis.monte_carlo_dgp import (
    dgp_interactive_fixed_effects_model_with_common_and_time_invariant,
)
from src.analysis.monte_carlo_dgp import dgp_time_invariant_fixed_effects_model
from src.analysis.simulation import simulation_coefficient
from src.analysis.simulation import statistics_coefficient


def test_simulation_coefficient_model1():
    dgp_fun = dgp_time_invariant_fixed_effects_model
    all_N = [10, 20]
    all_T = [9, 19]
    nsims = 2
    df_sim_result = simulation_coefficient(
        dgp_fun,
        all_N,
        all_T,
        nsims,
        need_sde=True,
        interactive_start_value_effect="pooling",
        within_effect="individual",
        beta1=1,
        beta2=3,
        mu=5,
        gamma=2,
        delta=4,
    )
    assert df_sim_result.shape == (4, 11)
    assert not df_sim_result.isna().any(axis=None, skipna=False)


def test_simulation_coefficient_model2():
    dgp_fun = dgp_additive_fixed_effects_model
    all_N = [10, 20]
    all_T = [9, 19]
    nsims = 2
    df_sim_result = simulation_coefficient(
        dgp_fun,
        all_N,
        all_T,
        nsims,
        need_sde=True,
        interactive_start_value_effect="twoways",
        within_effect="twoways",
        beta1=1,
        beta2=3,
        mu=5,
        gamma=2,
        delta=4,
    )
    assert df_sim_result.shape == (4, 11)
    assert not df_sim_result.isna().any(axis=None, skipna=False)


def test_simulation_coefficient_model3():
    dgp_fun = dgp_interactive_fixed_effects_model
    all_N = [10, 20]
    all_T = [9, 19]
    nsims = 2
    df_sim_result = simulation_coefficient(
        dgp_fun,
        all_N,
        all_T,
        nsims,
        need_sde=True,
        interactive_start_value_effect="pooling",
        within_effect="twoways",
        beta1=1,
        beta2=3,
        mu=5,
        gamma=2,
        delta=4,
    )
    assert df_sim_result.shape == (4, 15)
    assert not df_sim_result.isna().any(axis=None, skipna=False)


def test_simulation_coefficient_model4():
    dgp_fun = dgp_interactive_fixed_effects_model_with_common_and_time_invariant
    all_N = [10, 20]
    all_T = [9, 19]
    nsims = 2
    df_sim_result = simulation_coefficient(
        dgp_fun,
        all_N,
        all_T,
        nsims,
        need_sde=True,
        interactive_start_value_effect="pooling",
        within_effect="twoways",
        beta1=1,
        beta2=3,
        mu=5,
        gamma=2,
        delta=4,
    )
    assert df_sim_result.shape == (4, 23)
    na_expect = np.full(23, False)
    na_expect[[11, 12, 21, 22]] = True
    np.testing.assert_array_equal(
        df_sim_result.isna().any(axis=0, skipna=False), na_expect
    )


def test_statistics_coefficient():
    dgp_fun = dgp_interactive_fixed_effects_model_with_common_and_time_invariant
    all_N = [10, 20]
    all_T = [9, 19]
    nsims = 2
    df_sim_result = simulation_coefficient(
        dgp_fun,
        all_N,
        all_T,
        nsims,
        need_sde=True,
        interactive_start_value_effect="pooling",
        within_effect="twoways",
        beta1=1,
        beta2=3,
        mu=5,
        gamma=2,
        delta=4,
    )
    df_statistic = statistics_coefficient(
        all_N, all_T, nsims, df_sim_result, beta1=1, beta2=3, mu=5, gamma=2, delta=4
    )
    assert df_statistic.shape == (2, 62)
    na_expect = np.full(62, False)
    na_expect[[35, 36, 40, 41, 45, 46, 50, 51, 55, 56, 60, 61]] = True
    np.testing.assert_array_equal(
        df_statistic.isna().any(axis=0, skipna=False), na_expect
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[0, "mean_interactive.1"],
        df_sim_result.loc[0:1, "beta_interactive.1"].mean(axis=None),
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[1, "mean_interactive.1"],
        df_sim_result.loc[2:3, "beta_interactive.1"].mean(axis=None),
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[0, "bias_interactive.2"],
        abs(df_statistic.loc[0, "mean_interactive.2"] - 3) / 3,
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[1, "sde_within.1"],
        df_sim_result.loc[2:3, "sde_within.1"].mean(axis=None),
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[1, "ci_l_within.1"],
        df_statistic.loc[1, "mean_within.1"]
        - df_statistic.loc[1, "sde_within.1"] * 1.959963984540054,
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[1, "ci_u_within.1"],
        df_statistic.loc[1, "mean_within.1"]
        + df_statistic.loc[1, "sde_within.1"] * 1.959963984540054,
    )
    np.testing.assert_almost_equal(
        df_statistic.loc[0, "rmse_interactive.2"],
        np.sqrt(
            ((df_sim_result.loc[0:1, "beta_interactive.2"] - 3) ** 2).mean(axis=None)
        ),
    )
