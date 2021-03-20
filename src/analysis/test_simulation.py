import numpy as np

from src.analysis.monte_carlo_dgp import dgp_additive_fixed_effects_model
from src.analysis.monte_carlo_dgp import dgp_interactive_fixed_effects_model
from src.analysis.monte_carlo_dgp import (
    dgp_interactive_fixed_effects_model_with_common_and_time_invariant,
)
from src.analysis.monte_carlo_dgp import dgp_time_invariant_fixed_effects_model
from src.analysis.simulation import simulation_coefficient


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
    assert not np.isnan(df_sim_result).any(axis=None, skipna=False)


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
    assert not np.isnan(df_sim_result).any(axis=None, skipna=False)


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
    assert not np.isnan(df_sim_result).any(axis=None, skipna=False)


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
        np.isnan(df_sim_result).any(axis=0, skipna=False), na_expect
    )
