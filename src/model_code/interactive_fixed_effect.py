class InteractiveFixedEffect:
    r"""
    Interactive fixed effects estimator for panel data

    Parameters
    ----------
    dependent : array-like
        Dependent (left-hand-side) variable (time by entity).
    exog : array-like
        Exogenous or right-hand-side variables (variable by time by entity).

    Notes
    -----
    .. math::
        y_{it} = \beta x_{it} + \lambda_{i}'F_{t} + \epsilon_{it}
    """

    def __init__(self, dependent, exog):
        pass

    def fit(self, r, beta_hat_0=None, constant=True, tolerance=0.0001):
        """
        Estimate model parameters

        Parameters
        ----------
        r : int
            Number of factors.
        beta_hat_0: array-like, optional
            Starting values of estimator.
        constant : bool, optional
            Flag indicating to add constant.
        tolerance : float, optional
            Iteration precision.
        """
        pass
