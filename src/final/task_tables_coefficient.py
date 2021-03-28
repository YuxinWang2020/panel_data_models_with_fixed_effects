"""
Use tables to show the convergence properties of the estimators mentioned in the
paper, and compare how well they work in different panel data models.
"""
import json

import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on({"source": SRC / "final" / "tables_range_T_N.R"})
@pytask.mark.parametrize(
    "depends_on, produces, r",
    [
        (
            {
                "simulate": SRC / "model_specs" / f"{simulate_name}.json",
                "statistic": BLD / "analysis" / f"statistic_{simulate_name}.csv",
            },
            BLD / "tables" / f"table_{simulate_name}.csv",
            json.dumps(
                {
                    "simulate": str(SRC / "model_specs" / f"{simulate_name}.json"),
                    "statistic": str(
                        BLD / "analysis" / f"statistic_{simulate_name}.csv"
                    ),
                    "produces": str(BLD / "tables" / f"table_{simulate_name}.csv"),
                }
            ),
        )
        for simulate_name in [
            "range_T_N_model1",
            "range_T_N_model2",
            "range_T_N_model3",
            "range_T_N_model4",
        ]
    ],
)
def task_tables_range_T_N():
    pass
