"""
Use figures and tables to show the properties of factor estimation models in finite
panel data sets.
"""
import json

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.model_code.utils import paste0


@pytask.mark.depends_on(
    {
        "source": SRC / "final" / "figures_range_r.R",
        "simulate": SRC / "model_specs" / "range_r_model4.json",
        "sim_result": BLD / "analysis" / "sim_result_range_r.csv",
        "statistic": BLD / "analysis" / "statistic_range_r.csv",
    }
)
@pytask.mark.produces(list((BLD / "figures" / "range_r_model4").rglob("*.*")))
@pytask.mark.r(
    json.dumps(
        {
            "simulate": str(SRC / "model_specs" / "range_r_model4.json"),
            "sim_result": str(BLD / "analysis" / "sim_result_range_r.csv"),
            "statistic": str(BLD / "analysis" / "statistic_range_r.csv"),
            "out_dir": str(BLD / "figures" / "range_r_model4"),
        }
    )
)
def task_figures_range_r():
    pass


@pytask.mark.depends_on(BLD / "analysis" / "statistic_range_r.csv")
@pytask.mark.produces(BLD / "tables" / "table_range_r_model4.csv")
def task_tables_range_r(depends_on, produces):
    statistic_range_r = pd.read_csv(depends_on)
    startswith_vec = np.vectorize(str.startswith)
    p = sum(startswith_vec(statistic_range_r.columns, "mean_interactive."))
    coefficients = ["beta1", "beta2", "mu", "gamma", "delta"]
    select_statistics = {
        "colName": ["mean", "rmse", "sde"],
        "presentName": ["Mean", "SD", "SDE"],
    }

    table_ls = statistic_range_r.loc[
        :,
        ["r", "T", "N"]
        + paste0(
            select_statistics["colName"] * p,
            "_interactive.",
            np.repeat(range(1, p + 1), len(select_statistics["colName"])).tolist(),
        ),
    ]
    table_ls.set_axis(
        ["r", "T", "N"]
        + paste0(
            select_statistics["presentName"] * p,
            " ",
            np.repeat(coefficients[0:p], len(select_statistics["colName"])).tolist(),
        ),
        axis="columns",
        inplace=True,
    )
    table_ls.to_csv(produces, index=False)
