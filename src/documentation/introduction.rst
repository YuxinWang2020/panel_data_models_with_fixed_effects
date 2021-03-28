.. _introduction:


************
Introduction
************

In this project we present the main ideas of Bai (2009), Bai and Ng (2002), and Moon and Weidner (2015). We start with comparing the convergence properties and identification conditions for least squares estimators in panel data models, and then use econometric theory for factor model estimations in practice. We provide the algorithms in Bai (2009) and Bai, Ng (2002) in Python, replicate some Monte Carlo simulation results of the papers and investigate the simulation with other data generating processes and visualize them in R Language. In the end, we return to the previous empirical example and look at the estimation using the estimator proposed by Bai (2009).


.. _project structure:

Project structure
=================

The logic of this project works by step of the analysis:

1. Data management
2. Monte Carlo simulations and real data application
3. Visualisation and results formatting
4. Research paper and presentation

The main part of this project is Monte Carlo simulations and real data estimation. See section :ref:`monte_carlo_simulations_and_real_data_application` for more detail.


.. _getting_started:

Getting started
===============

**This assumes you have installed:**

* Miniconda or Anaconda
* a modern LaTeX distribution (e.g. TeXLive, MacTex, or MikTex)


1. Create the environment from the `environment.yml` file.

.. code-block:: console
    :linenos:

    $ conda env create -f environment.yml

2. Activate the newly created conda environment.

.. code-block:: console
    :linenos:

    $ conda activate panel_data_models_with_fixed_effects

3. Run pytask to start all tasks automaticly.

.. code-block:: console
    :linenos:

    $ conda develop .
    $ pytask

After that, all build result can be found in `bld` folder.

.. _built result :

Built result
============

The main analysis task takes hours to run. For quickly scan, some built result can be found in this repository:
`panel_data_models_build_result <https://github.com/YuxinWang2020/panel_data_models_build_result>`_
