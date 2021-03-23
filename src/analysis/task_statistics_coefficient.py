import json
import pickle

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
                "sim_result": BLD / "analysis" / f"sim_result_{simulate_name}.pickle",
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
    with open(depends_on["sim_result"], "rb") as read_file:
        simulate_in_sim_result, df_sim_result = pickle.load(read_file)
    # check if sim_result is updated after json is modified
    assert (
        simulate == simulate_in_sim_result
    ), f"Sim_result need to be updated. Params in sim_result:{simulate_in_sim_result}"

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
