import functools

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS
from scipy.stats import norm

from src.model_code.interactive_fixed_effect import InteractiveFixedEffect
from src.model_code.statistics import caculate_rmse
from src.model_code.utils import paste


def simulation_coefficient(
    dgp_func,
    all_N,
    all_T,
    nsims,
    *,
    need_sde=False,
    tolerance=0.0001,
    r=2,
    interactive_start_value_effect="pooling",
    within_effect="twoways",
    **beta_true
):
    """
    Monte carlo simulation to estimate beta hat and standard error of coefficient.

    Parameters
    ----------
    dgp_func : function
        One function in monte_carlo_dgp
    all_N : array-like
        Different sample sizes of entity
    all_T : array-like
        Different sample sizes of time
    nsims : int
        Simulation times under the same N and T
    need_sde : bool
        Flag the sde caculation conditions
    tolerance : float, optional
        Iteration precision.
    r : int, optional
        Number of factors.
    interactive_start_value_effect : string, optional
        The effects used in package linearmodels for starting value, one of "pooling",
        "twoways"
    within_effect : string, optional
        The effects used in package linearmodels for within estimator, one of "twoways",
        "individual"
    beta_true : float
        Coefficient of variables used in dgp_func. Values in ("beta1", "beta2", "mu",
        "gamma", "delta")

    Returns
    -------
    df_sim_result : DataFrame
        Columns are T, N, sim, *beta_interactive, *beta_within, *sde_interactive,
        *sde_within
    """

    assert len(all_N) == len(all_T), "all_N and all_T must has same length"
    assert interactive_start_value_effect in ("pooling", "twoways")
    assert within_effect in ("twoways", "individual")
    beta_true = {
        k: beta_true[k]
        for k in ("beta1", "beta2", "mu", "gamma", "delta")
        if k in beta_true
    }  # select only coefficient from beta_true
    dgp_func = functools.partial(dgp_func, **beta_true)  # set default arguments
    # Generate data frame of param.
    # It is Cartesian product of all_N and all_T and c(1:nsims).
    T_N_sim = pd.DataFrame(
        {
            "T": np.repeat(all_T, nsims),
            "N": np.repeat(all_N, nsims),
            "sim": np.tile(range(nsims), len(all_N)),
        }
    )
    df_sim_result = pd.DataFrame()
    for case in range(len(T_N_sim)):
        # gerate simulation data
        X, Y, panel_df = dgp_func(T=T_N_sim.loc[case, "T"], N=T_N_sim.loc[case, "N"])
        p = X.shape[0]
        # within model require no collinear variable combinations
        no_collinear_x_var = ["x" + str(i + 1) for i in range(min(p, 3))]
        # run estimator for starting value for interactive estimator
        if interactive_start_value_effect == "twoways":
            start_value_estimator = PanelOLS(
                panel_df.y,
                panel_df[no_collinear_x_var],
                entity_effects=True,
                time_effects=True,
            )
        else:
            start_value_estimator = PooledOLS(
                panel_df.y, panel_df[["x" + str(i) for i in range(1, p + 1)]]
            )
        start_value_result = start_value_estimator.fit()
        interactive_start_value = [
            *start_value_result.params.tolist(),
            *np.zeros(p - len(start_value_result.params)),
        ]
        # run interactive fixed effect estimator
        interactive_estimator = InteractiveFixedEffect(Y, X)
        (
            beta_hat_interactive,
            beta_hat_list,
            f_hat,
            lambda_hat,
        ) = interactive_estimator.fit(r, interactive_start_value, tolerance)
        # run within estimator with the same data
        if within_effect == interactive_start_value_effect:
            within_result = start_value_result
        elif within_effect == "individual":
            within_estimator = PanelOLS(
                panel_df.y, panel_df[no_collinear_x_var], entity_effects=True
            )
            within_result = within_estimator.fit()
        else:
            within_estimator = PanelOLS(
                panel_df.y,
                panel_df[no_collinear_x_var],
                entity_effects=True,
                time_effects=True,
            )
            within_result = within_estimator.fit()
        beta_hat_within = [
            *within_result.params.tolist(),
            *np.full(p - len(within_result.params), fill_value=np.nan),
        ]
        sde_within = [
            *within_result.std_errors.tolist(),
            *np.full(p - len(within_result.std_errors), fill_value=np.nan),
        ]
        # if need to caculate sde
        if need_sde:
            sde_interactive = interactive_estimator.calculate_sde(
                beta_hat_interactive, f_hat, lambda_hat
            )
            sde_interactive = np.sqrt(np.diag(sde_interactive))
        else:
            sde_interactive = np.full(shape=(p), fill_value=np.nan)
        one_sim_result = pd.DataFrame(
            [
                [
                    *T_N_sim.loc[case, ["T", "N", "sim"]],
                    *beta_hat_interactive,
                    *beta_hat_within,
                    *sde_interactive,
                    *sde_within,
                ]
            ],
            columns=[
                "T",
                "N",
                "sim",
                *paste("beta_interactive", range(1, p + 1), sep="."),
                *paste("beta_within", range(1, p + 1), sep="."),
                *paste("sde_interactive", range(1, p + 1), sep="."),
                *paste("sde_within", range(1, p + 1), sep="."),
            ],
        )
        df_sim_result = df_sim_result.append(one_sim_result)
    df_sim_result = df_sim_result.reset_index(drop=True)
    return df_sim_result


def statistics_coefficient(all_N, all_T, nsims, df_sim_result, **beta_true):
    """
    Generate statistics of each N & T, take the mean of different simulations, and
    store them in a data frame. We include mean, bias, the RMSE, standard error and
    cofidence interval in our statistical results.

    Parameters
    ----------
    all_N : array-like
        Different sample sizes of entity
    all_T : array-like
        Different sample sizes of time
    nsims : int
        Simulation times under the same N and T
    df_sim_result : DataFrame
        Simulation results from function `simulation_coefficient`
    beta_true : float
        Coefficients of variables used in dgp_func. Values in ("beta1", "beta2", "mu",
        "gamma", "delta")

    """
    # vectorize startswith() to apply it in a string list
    startswith_vec = np.vectorize(str.startswith)
    # guess number of variables from column names
    p = sum(startswith_vec(df_sim_result.columns, "beta_interactive."))
    assert len(beta_true) >= p, "short of beta_true"
    beta_true_list = [
        beta_true[k]
        for k in ("beta1", "beta2", "mu", "gamma", "delta")
        if k in beta_true
    ][:p]
    df_statistic = pd.DataFrame(
        index=range(len(all_N)),
        columns=[
            "T",
            "N",
            *paste("mean_interactive", range(1, p + 1), sep="."),
            *paste("bias_interactive", range(1, p + 1), sep="."),
            *paste("sde_interactive", range(1, p + 1), sep="."),
            *paste("ci_l_interactive", range(1, p + 1), sep="."),
            *paste("ci_u_interactive", range(1, p + 1), sep="."),
            *paste("rmse_interactive", range(1, p + 1), sep="."),
            *paste("mean_within", range(1, p + 1), sep="."),
            *paste("bias_within", range(1, p + 1), sep="."),
            *paste("sde_within", range(1, p + 1), sep="."),
            *paste("ci_l_within", range(1, p + 1), sep="."),
            *paste("ci_u_within", range(1, p + 1), sep="."),
            *paste("rmse_within", range(1, p + 1), sep="."),
        ],
    )

    # three quick function to get and modify df_statistic and df_sim_result

    def get_stat(col):
        return df_statistic.iloc[i, startswith_vec(df_statistic.columns, col)]

    def get_sim(col):
        return df_sim_result.iloc[
            row_range_df_sim, startswith_vec(df_sim_result.columns, col)
        ]

    def set_stat(col, value):
        df_statistic.iloc[i, startswith_vec(df_statistic.columns, col)] = value

    for i in range(len(all_N)):
        df_statistic.loc[i, "N"] = all_N[i]
        df_statistic.loc[i, "T"] = all_T[i]
        row_range_df_sim = range(i * nsims, (i + 1) * nsims)
        set_stat("mean_interactive", get_sim("beta_interactive").mean())
        set_stat("mean_within", get_sim("beta_within").mean())
        set_stat(
            "bias_interactive",
            get_stat("mean_interactive").sub(beta_true_list).abs().div(beta_true_list),
        )
        set_stat(
            "bias_within",
            get_stat("mean_within").sub(beta_true_list).abs().div(beta_true_list),
        )
        if not np.isnan(get_sim("sde_interactive")).all(axis=None, skipna=False):
            set_stat("sde_interactive", get_sim("sde_interactive").mean())
            set_stat(
                "ci_l_interactive",
                get_stat("mean_interactive").sub(
                    get_stat("sde_interactive").mul(norm.ppf(0.975)).values
                ),
            )
            set_stat(
                "ci_u_interactive",
                get_stat("mean_interactive").add(
                    get_stat("sde_interactive").mul(norm.ppf(0.975)).values
                ),
            )
        set_stat("sde_within", get_sim("sde_within").mean())
        set_stat(
            "ci_l_within",
            get_stat("mean_within").sub(
                get_stat("sde_within").mul(norm.ppf(0.975)).values
            ),
        )
        set_stat(
            "ci_u_within",
            get_stat("mean_within").add(
                get_stat("sde_within").mul(norm.ppf(0.975)).values
            ),
        )
        set_stat(
            "rmse_interactive",
            caculate_rmse(get_sim("beta_interactive"), beta_true_list),
        )
        set_stat("rmse_within", caculate_rmse(get_sim("beta_within"), beta_true_list))
    return df_statistic
