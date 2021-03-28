"""
Use figures and tables to show the properties of factor estimation models in finite
panel data sets.
"""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.model_code.utils import paste0


@pytask.mark.depends_on(
    {
        "twoways": BLD / "analysis" / "statistic_range_T_N_model2.csv",
        "pooling": BLD
        / "analysis"
        / "statistic_range_T_N_model2_pooling_start_value.csv",
    }
)
@pytask.mark.produces(BLD / "tables" / "start_value_model2.csv")
def task_tables_start_value(depends_on, produces):
    twoways = pd.read_csv(depends_on["twoways"])
    pooling = pd.read_csv(depends_on["pooling"])
    startswith_vec = np.vectorize(str.startswith)
    p = sum(startswith_vec(twoways.columns, "mean_interactive."))
    coefficients = ["beta1", "beta2", "mu", "gamma", "delta"]

    table_twoways = pd.concat(
        [
            pd.DataFrame({"starting value estimator": ["two-way"] * twoways.shape[0]}),
            twoways.loc[
                :,
                ["T", "N"]
                + paste0(
                    ["mean"] * p,
                    "_interactive.",
                    range(1, p + 1),
                ),
            ],
        ],
        axis=1,
    )
    table_pooling = pd.concat(
        [
            pd.DataFrame({"starting value estimator": ["pooled"] * pooling.shape[0]}),
            pooling.loc[
                :,
                ["T", "N"]
                + paste0(
                    ["mean"] * p,
                    "_interactive.",
                    range(1, p + 1),
                ),
            ],
        ],
        axis=1,
    )
    table_merge = pd.concat([table_pooling, table_twoways], axis=0)
    table_merge.set_axis(
        ["starting value estimator", "T", "N"] + coefficients[0:p],
        axis="columns",
        inplace=True,
    )
    table_merge.to_csv(produces, index=False)
