import numpy as np


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
        self._dependent = dependent
        self._exog = exog
        self._N = exog.shape[2]
        self._T = exog.shape[1]
        self._p = exog.shape[0]

    def fit(self, r, beta_hat_0=None, tolerance=0.0001):
        """
        Estimate model parameters

        Parameters
        ----------
        r : int
            Number of factors.
        beta_hat_0: array-like, optional
            Starting values of estimator. Should be same order as exog variable.
        tolerance : float, optional
            Iteration precision.

        Returns
        -------
        beta_hat : array-like
            Estimate result of slope coefficients. Same order as exog variable.
        beta_hat_list : array-like
            Iteration intermediate values.
        f_hat : array-like
            Estimate result of Factor.
        lambda_hat : array-like
            Estimate result or Lambda.
        """
        if beta_hat_0 is None:
            beta_hat_0 = np.zeros(shape=(1, self._p))
        else:
            beta_hat_0 = np.array(beta_hat_0).reshape(1, self._p)
        beta_hat_list = beta_hat_0
        e = np.inf
        while e > tolerance:
            f_hat = self._calculate_f_hat(beta_hat_0, r)
            lambda_hat = self._calculate_lambda_hat(beta_hat_0, f_hat, r)
            beta_hat = self._calculate_beta_hat(f_hat, lambda_hat)
            beta_hat_list = np.row_stack((beta_hat_list, beta_hat))
            e = np.linalg.norm(beta_hat - beta_hat_0, ord=2)
            beta_hat_0 = beta_hat
        return (beta_hat, beta_hat_list, f_hat, lambda_hat)

    def _calculate_f_hat(self, beta_hat, r):
        wwt = np.zeros(shape=(self._T, self._T))
        for i in range(self._N):
            w_i = self._dependent[:, i] - beta_hat.dot(self._exog[:, :, i])
            wwt = wwt + w_i.T.dot(w_i)
        w, v = np.linalg.eig(wwt)
        f_hat = np.sqrt(self._T) * v[:, np.argsort(-w)[0:r]]
        return f_hat

    def _calculate_lambda_hat(self, beta_hat, f_hat, r):
        lambda_hat = np.full(shape=(self._N, r), fill_value=np.nan)
        for i in range(self._N):
            lambda_hat[i, :] = (
                self._dependent[:, i] - beta_hat.dot(self._exog[:, :, i])
            ).dot(f_hat) / self._T
        return lambda_hat

    def _calculate_beta_hat(self, f_hat, lambda_hat):
        A = np.zeros(shape=(self._p, self._p))
        B = np.zeros(shape=(1, self._p))
        for i in range(self._N):
            A = A + self._exog[:, :, i].dot(self._exog[:, :, i].T)
            B = B + self._exog[:, :, i].dot(
                (self._dependent[:, i] - f_hat.dot(lambda_hat[i, :])).T
            )
        beta_hat = B.dot(np.linalg.inv(A))
        return beta_hat
