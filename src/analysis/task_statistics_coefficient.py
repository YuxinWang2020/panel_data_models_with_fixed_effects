"""
Task for caculate statistical results for coefficients by using `statistics_coefficient`
function. The inputs of the function come from `task_simulation_coefficient`.
"""
import json

import pandas as pd
import pytask

from src.analysis.simulation import statistics_coefficient
from src.config import BLD
from src.config import SRC


@pytask.mark.parametrize(
    "depends_on, produces",
    [
        (
            {
                "simulate": SRC / "model_specs" / f"{simulate_name}.json",
                "sim_result": BLD / "analysis" / f"sim_result_{simulate_name}.csv",
            },
            BLD / "analysis" / f"statistic_{simulate_name}.csv",
        )
        for simulate_name in [
            "range_N_model1",
            "range_N_model2",
            "range_N_model3",
            "range_N_model4",
            "range_T_N_model1",
            "range_T_N_model2",
            "range_T_N_model3",
            "range_T_N_model4",
            "range_grid_T_N_model1",
            "range_grid_T_N_model2",
            "range_grid_T_N_model3",
            "range_grid_T_N_model4",
        ]
    ],
)
def task_statistics_coefficient(depends_on, produces):
    simulate = json.loads(depends_on["simulate"].read_text(encoding="utf-8"))
    df_sim_result = pd.read_csv(depends_on["sim_result"])

    # Run the monte carlo simulation
    df_statistic = statistics_coefficient(
        all_N=simulate["all_N"],
        all_T=simulate["all_T"],
        nsims=simulate["nsims"],
        df_sim_result=df_sim_result,
        **simulate["beta_true"],
    )
    # Store df_statistic with locations after each round
    df_statistic.to_csv(produces, index=False)
