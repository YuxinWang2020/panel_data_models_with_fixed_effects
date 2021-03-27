.. _analysis:

.. _monte_carlo_simulations_and_real_data_application

*************************************************
Monte Carlo simulations and real data application
*************************************************

Documentation of the code in *src.analysis*. This is the core of the project.

The analysis in this project contains three aspects:

1. Estimation of slope coefficients
    1. IID data simulations
    2. ARI data simulations
    3. Starting Values in additive fixed effects model
2. Estimation of the number of factors
    1. Estimate interactive fixed effects model for different number of factors
    2. Estimate factor number by choosing different penalty functions with the criteria PC and IC
3. Real data application


Process of Monte Carlo simulations:

1. Monte carlo data generating processes
2. Interactive-effects estimator and within estimator
3. Statistics of mean, bias, rmse, standard error and cofidence interval


Data Generating Processor for Monte Carlo Simulation
====================================================

.. automodule:: src.analysis.monte_carlo_dgp
    :members:


Monte Carlo Simulation
====================================================

.. automodule:: src.analysis.simulation
    :members:


Estimation of slope coefficients
====================================================

.. automodule:: src.analysis.task_simulation_coefficient
    :members:


Statistics
====================================================

.. automodule:: src.analysis.task_statistics_coefficient
    :members:


Estimation of the number of factors
====================================================

.. automodule:: src.analysis.task_simulation_factor
    :members:
