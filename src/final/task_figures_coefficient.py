import json

import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on({"source": SRC / "final" / "figures_range_N.R"})
@pytask.mark.parametrize(
    "depends_on, produces, r",
    [
        (
            {
                "simulate": SRC / "model_specs" / f"{simulate_name}.json",
                "sim_result": BLD / "analysis" / f"sim_result_{simulate_name}.csv",
            },
            list((BLD / "figures" / simulate_name).rglob("*.*")),
            json.dumps(
                {
                    "simulate": str(SRC / "model_specs" / f"{simulate_name}.json"),
                    "sim_result": str(
                        BLD / "analysis" / f"sim_result_{simulate_name}.csv"
                    ),
                    "out_dir": str(BLD / "figures" / simulate_name),
                }
            ),
        )
        for simulate_name in [
            "range_N_model1",
            "range_N_model2",
            "range_N_model3",
            "range_N_model4",
        ]
    ],
)
def task_figures_range_N():
    pass


@pytask.mark.depends_on({"source": SRC / "final" / "figures_range_grid_T_N.R"})
@pytask.mark.parametrize(
    "depends_on, produces, r",
    [
        (
            {
                "simulate": SRC / "model_specs" / f"{simulate_name}.json",
                "statistic": BLD / "analysis" / f"statistic_{simulate_name}.csv",
            },
            list((BLD / "figures" / simulate_name).rglob("*.*")),
            json.dumps(
                {
                    "simulate": str(SRC / "model_specs" / f"{simulate_name}.json"),
                    "statistic": str(
                        BLD / "analysis" / f"statistic_{simulate_name}.csv"
                    ),
                    "out_dir": str(BLD / "figures" / simulate_name),
                }
            ),
        )
        for simulate_name in [
            "range_grid_T_N_model1",
            "range_grid_T_N_model2",
            "range_grid_T_N_model3",
            "range_grid_T_N_model4",
        ]
    ],
)
def task_figures_range_grid_T_N():
    pass
