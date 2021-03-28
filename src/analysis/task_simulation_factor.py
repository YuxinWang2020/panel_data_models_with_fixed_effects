"""
Task for estimating the parameters in interactive fixed effects model by using different
numbers of factors.
Task for estimating factor numbers in interactive fixed effects model by choosing
different penalty functions with criteria PC and IC.
"""
import json

import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PooledOLS

from src.analysis.monte_carlo_dgp import (
    dgp_interactive_fixed_effects_model_with_common_and_time_invariant,
)
from src.analysis.monte_carlo_dgp import dgp_random_iid_residual
from src.analysis.simulation import simulation_coefficient
from src.analysis.simulation import statistics_coefficient
from src.config import BLD
from src.config import SRC
from src.model_code.factor_estimator import FactorEstimator
from src.model_code.interactive_fixed_effect import InteractiveFixedEffect


@pytask.mark.skip
@pytask.mark.depends_on(SRC / "model_specs" / "range_r_model4.json")
@pytask.mark.produces(
    {
        "sim_result": BLD / "analysis" / "sim_result_range_r.csv",
        "statistic": BLD / "analysis" / "statistic_range_r.csv",
    }
)
def task_simulation_and_statistics_range_r(depends_on, produces):
    """
    Task for estimating the parameters in interactive fixed effects model by using
    different numbers of factors.
    """
    # set parameters
    simulate = json.loads(depends_on.read_text(encoding="utf-8"))
    dgp_func = globals()[simulate["dgp_func"]]
    all_r = simulate["all_r"]
    np.random.seed(simulate["rng_seed"])
    # Run the monte carlo simulation
    df_sim_range_r = pd.DataFrame()
    df_statistic_range_r = pd.DataFrame()
    for r in all_r:
        df_sim_result = simulation_coefficient(
            dgp_func=dgp_func,
            all_N=simulate["all_N"],
            all_T=simulate["all_T"],
            nsims=simulate["nsims"],
            need_sde=simulate["need_sde"],
            tolerance=simulate["tolerance"],
            r=r,
            interactive_start_value_effect=simulate["interactive_start_value_effect"],
            within_effect=simulate["within_effect"],
            **simulate["beta_true"],
        )
        df_statistic = statistics_coefficient(
            all_N=simulate["all_N"],
            all_T=simulate["all_T"],
            nsims=simulate["nsims"],
            df_sim_result=df_sim_result,
            **simulate["beta_true"],
        )
        df_sim_range_r = df_sim_range_r.append(
            pd.concat(
                [pd.DataFrame({"r": [r] * df_sim_result.shape[0]}), df_sim_result],
                axis=1,
            )
        )
        df_statistic_range_r = df_statistic_range_r.append(
            pd.concat(
                [pd.DataFrame({"r": [r] * df_statistic.shape[0]}), df_statistic], axis=1
            )
        )
    # Store df_sim_range_r and df_statistic_range_r
    df_sim_range_r = df_sim_range_r.reset_index(drop=True)
    df_sim_range_r.to_csv(produces["sim_result"], index=False)
    df_statistic_range_r = df_statistic_range_r.reset_index(drop=True)
    df_statistic_range_r.to_csv(produces["statistic"], index=False)


@pytask.mark.produces(BLD / "tables" / "determine_num_of_factors_random_iid.csv")
def task_factor_estimate_random_iid_residual(produces):
    """
    Task for estimating factor numbers in the model defined by the function
    `dgp_random_iid_residual`.
    We choose different penalty functions g1,g2,g3 with criterias PC and IC.
    It replicates the result of Table 2 in Bai,Ng (2002), page 205.
    """
    r = 3
    rmax = 8
    nsims = 1000
    all_N = [100, 100, 200, 500, 1000]
    all_T = [40, 60, 60, 60, 60]
    df_factor_estimate = pd.DataFrame()
    np.random.seed(123)
    for case in range(len(all_N)):
        N = all_N[case]
        T = all_T[case]
        df_sim = pd.DataFrame(
            index=range(nsims),
            columns=["T", "N", "PC1", "PC2", "PC3", "IC1", "IC2", "IC3"],
        )
        df_sim["T"] = [T] * nsims
        df_sim["N"] = [N] * nsims
        for i in range(nsims):
            residual = dgp_random_iid_residual(N, T, r)
            factor_estimator = FactorEstimator(residual)
            df_sim.loc[i, "PC1"] = factor_estimator.r_hat(rmax, "PC", 1)
            df_sim.loc[i, "PC2"] = factor_estimator.r_hat(rmax, "PC", 2)
            df_sim.loc[i, "PC3"] = factor_estimator.r_hat(rmax, "PC", 3)
            df_sim.loc[i, "IC1"] = factor_estimator.r_hat(rmax, "IC", 1)
            df_sim.loc[i, "IC2"] = factor_estimator.r_hat(rmax, "IC", 2)
            df_sim.loc[i, "IC3"] = factor_estimator.r_hat(rmax, "IC", 3)
        df_factor_estimate = df_factor_estimate.append(
            pd.DataFrame(df_sim.mean(axis=0)).T
        )
    df_factor_estimate = df_factor_estimate.reset_index(drop=True)
    df_factor_estimate.to_csv(produces, index=False)


@pytask.mark.produces(BLD / "tables" / "determine_num_of_factors_model4.csv")
def task_factor_estimate_interactive_fixed_effects_model(produces):
    """
    Task for estimating factor numbers in interactive fixed effects model.
    We choose different penalty functions g1,g2,g3 with criterias PC and IC.
    """
    rmax = 8
    nsims = 1000
    all_N = [100, 100, 100, 100, 10, 20, 50]
    all_T = [10, 20, 50, 100, 100, 100, 100]
    dgp_func = dgp_interactive_fixed_effects_model_with_common_and_time_invariant
    tolerance = 0.0001
    beta_true = {"beta1": 1, "beta2": 3, "mu": 5, "gamma": 2, "delta": 4}
    r0 = 8
    df_factor_estimate = pd.DataFrame()
    np.random.seed(123)
    for case in range(len(all_N)):
        N = all_N[case]
        T = all_T[case]
        df_sim = pd.DataFrame(
            index=range(nsims),
            columns=["T", "N", "PC1", "PC2", "PC3", "IC1", "IC2", "IC3"],
        )
        df_sim["T"] = [T] * nsims
        df_sim["N"] = [N] * nsims
        for i in range(nsims):
            X, Y, panel_df = dgp_func(T, N, **beta_true)
            start_value_estimator = PooledOLS(
                panel_df.y, panel_df[["x" + str(i) for i in range(1, 6)]]
            )
            start_value_result = start_value_estimator.fit()
            interactive_start_value = start_value_result.params.tolist()
            interactive_estimator = InteractiveFixedEffect(Y, X)
            beta_hat, beta_hat_list, f_hat, lambda_hat = interactive_estimator.fit(
                r0, interactive_start_value, tolerance
            )
            residual = Y - (X.T.dot(beta_hat)).T
            factor_estimator = FactorEstimator(residual)
            df_sim.loc[i, "PC1"] = factor_estimator.r_hat(rmax, "PC", 1)
            df_sim.loc[i, "PC2"] = factor_estimator.r_hat(rmax, "PC", 2)
            df_sim.loc[i, "PC3"] = factor_estimator.r_hat(rmax, "PC", 3)
            df_sim.loc[i, "IC1"] = factor_estimator.r_hat(rmax, "IC", 1)
            df_sim.loc[i, "IC2"] = factor_estimator.r_hat(rmax, "IC", 2)
            df_sim.loc[i, "IC3"] = factor_estimator.r_hat(rmax, "IC", 3)
        df_factor_estimate = df_factor_estimate.append(
            pd.DataFrame(df_sim.mean(axis=0)).T
        )
    df_factor_estimate = df_factor_estimate.reset_index(drop=True)
    df_factor_estimate.to_csv(produces, index=False)
