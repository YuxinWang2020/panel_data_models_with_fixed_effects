import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS

from src.config import BLD
from src.model_code.factor_estimator import FactorEstimator
from src.model_code.interactive_fixed_effect import InteractiveFixedEffect


@pytask.mark.depends_on(BLD / "data" / "cigar_clean.csv")
@pytask.mark.produces(
    {
        "within": BLD / "tables" / "real_data_within.csv",
        "interactive": BLD / "tables" / "real_data_interactive.csv",
    }
)
def task_real_data_coefficient(depends_on, produces):
    cigar_clean = pd.read_csv(depends_on)
    cigar_clean = cigar_clean.set_index(["state", "year"])
    cigar_clean.insert(0, "Intercept", [1] * cigar_clean.shape[0])  # add Intercept
    # OLS estimator
    pool_estimator = PooledOLS(
        cigar_clean["log_C_it"],
        cigar_clean[
            ["Intercept", "log_C_it_lag1", "log_P_it", "log_Pn_it", "log_Y_it"]
        ],
    )
    pool_result = pool_estimator.fit()
    pool_beta_hat = pool_result.params[1:]  # drop Intercept
    # within estimator
    within_estimator = PanelOLS(
        cigar_clean["log_C_it"],
        cigar_clean[
            ["Intercept", "log_C_it_lag1", "log_P_it", "log_Pn_it", "log_Y_it"]
        ],
        entity_effects=True,
        time_effects=True,
    )
    within_result = within_estimator.fit()
    within_beta_hat = within_result.params[1:]  # drop Intercept
    # save result to output
    within_df = pd.DataFrame({"OLS": pool_beta_hat, "Within": within_beta_hat}).T
    within_df.to_csv(produces["within"], index=True)
    # interactive estimator
    tolerance = 0.001
    states = cigar_clean.index.to_frame()["state"].drop_duplicates().to_list()
    year = cigar_clean.index.to_frame()["year"].drop_duplicates().to_list()
    X = np.full((5, len(year), len(states)), np.nan)  # (variable by time by entity)
    Y = np.full((len(year), len(states)), np.nan)  # (time by entity)
    for i in range(len(states)):
        X[:, :, i] = cigar_clean.loc[
            (states[i], slice(None)),
            ["Intercept", "log_C_it_lag1", "log_P_it", "log_Pn_it", "log_Y_it"],
        ].T
        Y[:, i] = cigar_clean.loc[(states[i], slice(None)), "log_C_it"]
    interactive_start_value = pool_result.params.to_list()
    interactive_estimator = InteractiveFixedEffect(Y, X)
    all_r = list(range(1, 11))
    df_range_r = pd.DataFrame(
        {
            "Factor number": [0] + all_r,
            "Intercept": np.nan,
            "log_C_it_lag1": np.nan,
            "log_P_it": np.nan,
            "log_Pn_it": np.nan,
            "log_Y_it": np.nan,
        }
    )
    df_range_r.iloc[0, 1:] = pool_result.params.to_list()
    for i in range(len(all_r)):
        (
            interactive_beta_hat,
            beta_hat_list,
            f_hat,
            lambda_hat,
        ) = interactive_estimator.fit(all_r[i], interactive_start_value, tolerance)
        df_range_r.iloc[i + 1, 1:] = interactive_beta_hat
    # save result to output
    df_range_r.to_csv(produces["interactive"], index=False)


@pytask.mark.depends_on(
    {
        "cigar_clean": BLD / "data" / "cigar_clean.csv",
        "interactive": BLD / "tables" / "real_data_interactive.csv",
    }
)
@pytask.mark.produces(BLD / "tables" / "real_data_num_of_factors.csv")
def task_factor_estimate_real_data(depends_on, produces):
    cigar_clean = pd.read_csv(depends_on["cigar_clean"])
    cigar_clean = cigar_clean.set_index(["state", "year"])
    cigar_clean.insert(0, "Intercept", [1] * cigar_clean.shape[0])  # add Intercept
    states = cigar_clean.index.to_frame()["state"].drop_duplicates().to_list()
    year = cigar_clean.index.to_frame()["year"].drop_duplicates().to_list()
    X = np.full((5, len(year), len(states)), np.nan)  # (variable by time by entity)
    Y = np.full((len(year), len(states)), np.nan)  # (time by entity)
    for i in range(len(states)):
        X[:, :, i] = cigar_clean.loc[
            (states[i], slice(None)),
            ["Intercept", "log_C_it_lag1", "log_P_it", "log_Pn_it", "log_Y_it"],
        ].T
        Y[:, i] = cigar_clean.loc[(states[i], slice(None)), "log_C_it"]

    rmax = 10
    r0s = list(range(1, 11))
    df_r_hat = pd.DataFrame(
        {
            "R": r0s,
            "PC1": np.nan,
            "PC2": np.nan,
            "PC3": np.nan,
            "IC1": np.nan,
            "IC2": np.nan,
            "IC3": np.nan,
        }
    )
    interactive_beta_hat = pd.read_csv(depends_on["interactive"])
    for i in range(len(r0s)):
        beta_hat = interactive_beta_hat[
            interactive_beta_hat["Factor number"] == r0s[i]
        ].iloc[0, 1:]
        residual = Y - (X.T.dot(beta_hat)).T
        factor_estimator = FactorEstimator(residual)
        df_r_hat.loc[i, "PC1"] = factor_estimator.r_hat(rmax, "PC", 1)
        df_r_hat.loc[i, "PC2"] = factor_estimator.r_hat(rmax, "PC", 2)
        df_r_hat.loc[i, "PC3"] = factor_estimator.r_hat(rmax, "PC", 3)
        df_r_hat.loc[i, "IC1"] = factor_estimator.r_hat(rmax, "IC", 1)
        df_r_hat.loc[i, "IC2"] = factor_estimator.r_hat(rmax, "IC", 2)
        df_r_hat.loc[i, "IC3"] = factor_estimator.r_hat(rmax, "IC", 3)

    df_r_hat.to_csv(produces, index=False)
