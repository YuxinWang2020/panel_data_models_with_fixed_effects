"""
Some pure functions to caculate statistical results
"""
import numpy as np


def caculate_rmse(estimated, real):
    """
    Caculate the root-mean-square-error

    Parameters
    ----------
    estimated : array-like
        Estimated value
    real : array-like
        Real value
    """
    real = np.array(real)
    estimated = np.array(estimated)
    assert real.ndim == 1, "real value must be vector"
    if estimated.ndim == 1:
        estimated = np.array([estimated])
    assert estimated.shape[1] == len(
        real
    ), "estimated value and real value should have same length"
    return np.sqrt(((estimated - real) ** 2).mean(axis=0))


def caculate_sde(estimated):
    """
    Caculate standard error

    Parameters
    ----------
    estimated : array-like
        Estimated value
    """
    if np.array(estimated).ndim == 1:
        return np.zeros_like(estimated)
    return np.std(estimated, axis=0) / np.sqrt(len(estimated))
