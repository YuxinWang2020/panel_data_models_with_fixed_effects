.. _monte_carlo_simulations_and_real_data_application:

*************************************************
Monte Carlo Simulations and Real Data Application
*************************************************

Documentation of the code in *src.analysis*. This is the core of the project.

The analysis in this project contains three aspects:

1. Estimation of slope coefficients
    1. Monte carlo simulations under IID assumptions
    2. Monte carlo simulations under ARI time fixed effects
    3. Starting Values in the additive fixed effects model
2. Estimation of the number of factors
    1. Estimate the parameters in interactive fixed effects model by using different numbers of factors
    2. Estimate factor numbers in interactive fixed effects model by choosing different penalty functions and criterias
3. Real data application
    1. Use a panel data set of 46 observations from 1963 to 1992
    2. Estimate the coefficients and factor numbers by using the model in Baltagi and Levin (1992)


Process of Monte Carlo simulations:

1. Data generating processes for monte carlo simulations
2. Estimate model parameters by interactive-effects estimator and within estimator
3. Caculate statistical results for coefficients, including mean, bias, root-mean-square error, standard error and confidence interval

Function `simulation_coefficient` in module `src.analysis.simulation` is defined for running the simulations under different number of individuals and time periods over 1000 repetitions.


Data Generating process for Monte Carlo Simulation
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
