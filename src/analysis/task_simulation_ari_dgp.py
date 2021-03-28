import json

import numpy as np
import pytask

""" These three functions are used via `globals()`. They are not unused functions. """
from src.analysis.monte_carlo_dgp import (  # noqa: F401
    dgp_additive_fixed_effects_model_no_iid,
)
from src.analysis.monte_carlo_dgp import (  # noqa: F401
    dgp_interactive_fixed_effects_model_no_iid,
)
from src.analysis.monte_carlo_dgp import (  # noqa: F401
    dgp_interactive_fixed_effects_model_with_common_and_time_invariant_no_iid,
)
from src.analysis.simulation import simulation_coefficient
from src.config import BLD
from src.config import SRC


@pytask.mark.parametrize(
    "depends_on, produces",
    [
        (
            SRC / "model_specs" / f"{simulate_name}.json",
            BLD / "analysis" / f"sim_result_{simulate_name}.csv",
        )
        for simulate_name in [
            "range_T_N_model2_no_iid",
            "range_T_N_model3_no_iid",
            "range_T_N_model4_no_iid",
        ]
    ],
)
def task_simulation_ari_dgp(depends_on, produces):
    simulate = json.loads(depends_on.read_text(encoding="utf-8"))

    np.random.seed(simulate["rng_seed"])
    dgp_func = globals()[simulate["dgp_func"]]
    # Run the monte carlo simulation
    df_sim_result = simulation_coefficient(
        dgp_func=dgp_func,
        all_N=simulate["all_N"],
        all_T=simulate["all_T"],
        nsims=simulate["nsims"],
        need_sde=simulate["need_sde"],
        tolerance=simulate["tolerance"],
        r=simulate["r"],
        interactive_start_value_effect=simulate["interactive_start_value_effect"],
        within_effect=simulate["within_effect"],
        **simulate["beta_true"],
    )
    # Store df_sim_result with locations after each round
    df_sim_result.to_csv(produces, index=False)
