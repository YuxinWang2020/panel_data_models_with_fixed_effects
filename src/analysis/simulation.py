import functools

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS

from src.model_code.interactive_fixed_effect import InteractiveFixedEffect
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
    Monte carlo simulation to estimate beta hat and sde of coefficient.

    Parameters
    ----------
    dgp_func : function
        One function in monte_carlo_dgp
    all_N : array-like
        All sample size of entity
    all_T : array-like
        All sample size of time
    nsims : int
        Simulation times for the same N and T
    need_sde : bool
        Flag whether to caculate sde, which takes lots of times
    tolerance : float, optional
        Iteration precision.
    r : int, optional
        Number of factors.
    interactive_start_value_effect : string, optional
        The effects used in linearmodels for starting value, one of "pooling", "twoways"
    within_effect : string, optional
        The effects used in linearmodels for within estimate, one of "twoways",
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
