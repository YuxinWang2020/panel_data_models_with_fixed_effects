import profile

import numpy as np

from src.analysis.monte_carlo_dgp import (
    dgp_interactive_fixed_effects_model_with_common_and_time_invariant,
)
from src.analysis.simulation import simulation_coefficient  # noqa:F401

np.random.seed(123)
dgp_func = dgp_interactive_fixed_effects_model_with_common_and_time_invariant
all_N = [10, 20, 50]
all_T = [10, 20, 50]
nsims = 4
need_sde = True
tolerance = 0.01
r = 2
interactive_start_value_effect = "pooling"
within_effect = "twoways"
beta_true = {"beta1": 1, "beta2": 3, "mu": 5, "gamma": 2, "delta": 4}
# data = simulation_coefficient(
#     dgp_func,
#     all_N,
#     all_T,
#     nsims,
#     need_sde=need_sde,
#     tolerance=tolerance,
#     r=r,
#     interactive_start_value_effect=interactive_start_value_effect,
#     within_effect=within_effect,
#     **beta_true
# )
# print(data)
if __name__ == "__main__":
    profile.run(
        """simulation_coefficient(
        dgp_func,
        all_N,
        all_T,
        nsims,
        need_sde=need_sde,
        tolerance=tolerance,
        r=r,
        interactive_start_value_effect=interactive_start_value_effect,
        within_effect=within_effect,
        **beta_true
    )"""
    )
