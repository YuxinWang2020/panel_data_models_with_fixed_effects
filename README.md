## Introduction

In this project we present the main ideas of Bai (2009), Bai and Ng (2002), and Moon and Weidner (2015). We start with comparing the convergence properties and identification conditions for least squares estimators in panel data models, and then use econometric theory for factor model estimations in practice. We provide the algorithms in Bai (2009) and Bai, Ng (2002) in Python, replicate some Monte Carlo simulation results of the papers and investigate the simulation with other data generating processes and visualize them in R Language. In the end, we return to the previous empirical example and look at the estimation using the estimator proposed by Bai (2009).


## Project structure

The logic of this project works by step of the analysis:

1. Data management
2. Monte Carlo simulations and real data application
3. Visualisation and results formatting
4. Research paper and presentations


## Monte Carlo Simulations and Real Data Application

### The analysis in this project contains three aspects

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


### Process of Monte Carlo simulations:

1. Data generating processes for monte carlo simulations
2. Estimate model parameters by interactive-effects estimator and within estimator
3. Caculate statistical results for coefficients, including mean, bias, root-mean-square error, standard error and confidence interval

Function `simulation_coefficient` in module `src.analysis.simulation` is defined for running the simulations under different number of individuals and time periods over 1000 repetitions.



## Getting started

**This assumes you have installed:**
- Miniconda or Anaconda
- a modern LaTeX distribution (e.g. TeXLive, MacTex, or MikTex)


1. Create the environment from the `environment.yml` file.

``` shell
    $ conda env create -f environment.yml
```

2. Activate the newly created conda environment.

``` shell
    $ conda activate panel_data_models_with_fixed_effects
```

3. Run pytask to start all tasks automaticly.

``` shell
    $ conda develop .
    $ pytask
```
After that, all build result can be found in `bld` folder.


## Built result

The main analysis task takes hours to run. For quickly scan, some built result can be found in this repository:
[panel_data_models_build_result](https://github.com/YuxinWang2020/panel_data_models_build_result)
