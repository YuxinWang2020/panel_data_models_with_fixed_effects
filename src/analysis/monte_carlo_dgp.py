import numpy as np


def dgp_time_invariant_fixed_effects_model(T, N, *, beta1, beta2):
    r"""
    Monte carlo data generate processor for "Interactive Fixed Effects Model with Common
    Regressors and Time-invariant Regressors"

    Parameters
    ----------
    T : int
        Sample size of time
    N : int
        Sample size of entity
    beta1 : float
        Coefficient of x1
    beta2 : float
        Coefficient of x2

    Returns
    -------
    X : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y : array-like
        Simulate data of Dependent (left-hand-side) variable (time by entity).

    Notes
    -----
    .. math::
        y_{it} = \beta_{1}x_{it,1}+\beta_{2}x_{it,2}+\alpha_{i}+\epsilon_{it}
    The regressors are generated according to :math:`X_{it,j}=3+2\alpha_i+\eta_{it,j}`,
    with
    .. math::
        \eta_{it,j}\stackrel{\text{i.i.d}}{\sim} N(0,1), \qquad j\in \{1,2\},
        \alpha_i\stackrel{\text{i.i.d}}{\sim} N(0,1),
        \epsilon_{it}\stackrel{\text{i.i.d}}{\sim} N(0,4).
    """
    # Set parameters
    p = 2
    mu = 0
    gamma = 0
    delta = 0
    factor = np.tile([[1], [0]], (1, T))
    lambda_ = np.vstack(
        (np.random.normal(loc=0, scale=1, size=(1, N)), np.ones(shape=(1, N)))
    )
    X, Y = _dgp_fixed_effect_panel_data(
        T, N, beta1, beta2, mu, gamma, delta, factor, lambda_
    )
    X = X[0:p, :, :]
    return X, Y


def dgp_additive_fixed_effects_model(T, N, *, beta1, beta2):
    r"""
    Monte carlo data generate processor for "Interactive Fixed Effects Model with Common
    Regressors and Time-invariant Regressors"

    Parameters
    ----------
    T : int
        Sample size of time
    N : int
        Sample size of entity
    beta1 : float
        Coefficient of x1
    beta2 : float
        Coefficient of x2

    Returns
    -------
    X : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y : array-like
        Simulate data of Dependent (left-hand-side) variable (time by entity).

    Notes
    -----
    .. math::
        y_{it} = \beta_{1}x_{it,1}+\beta_{2}x_{it,2}+\alpha_{i}+\xi_{t}+\epsilon_{it}
    Two fixed effects satisfy
    .. math::
        \alpha_{i}, \xi_{t}\stackrel{\text{i.i.d}}{\sim}N(0,1).
    Both of them are correlated with the two regressors
    .. math::
        \[
            X_{it,j}=3+2\alpha_i+2\xi_t+\eta_{it,j},
        \]
    with
    .. math::
        \[
            \eta_{it,j}\stackrel{\text{i.i.d}}{\sim} N(0,1) \qquad j\in \{1,2\}.
        \]
    The regression error
    .. math::
        \epsilon_{it} \stackrel{\text{i.i.d}}{\sim}N(0,4).
    """
    # Set parameters
    p = 2
    mu = 0
    gamma = 0
    delta = 0
    factor = np.vstack(
        (np.ones(shape=(1, T)), np.random.normal(loc=0, scale=1, size=(1, T)))
    )
    lambda_ = np.vstack(
        (np.random.normal(loc=0, scale=1, size=(1, N)), np.ones(shape=(1, N)))
    )
    X, Y = _dgp_fixed_effect_panel_data(
        T, N, beta1, beta2, mu, gamma, delta, factor, lambda_
    )
    X = X[0:p, :, :]
    return X, Y


def dgp_interactive_fixed_effects_model(T, N, *, beta1, beta2, mu):
    r"""
    Monte carlo data generate processor for "Interactive Fixed Effects Model"

    Parameters
    ----------
    T : int
        Sample size of time
    N : int
        Sample size of entity
    beta1 : float
        Coefficient of x1
    beta2 : float
        Coefficient of x2
    mu : float
        Constant

    Returns
    -------
    X : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y : array-like
        Simulate data of Dependent (left-hand-side) variable (time by entity).

    Notes
    -----
    .. math::
        y_{it} = \beta_{1}x_{it,1}+\beta_{2}x_{it,2}+\mu+\lambda_{i}'F_{t}+\epsilon_{it}
    where
    .. math::
    \[
        \lambda_i = \binom{\lambda_{i1}}{\lambda_{i2}} \stackrel{\text{i.i.d}}{\sim}
        N(0,I_2),
    \]
    \[
        F_t =\binom{F_{t1}}{F_{t2}}\stackrel{\text{i.i.d}}{\sim}  N(0,I_2),
    \]
    The regressors are generated according to:
    .. math::
        X_{it,j}= 1+\lambda_{i1}F_{t1}+\lambda_{i2}F_{t2}+\lambda_{i1}+\lambda_{i2}+
        F_{t1}+F_{t2}+\eta_{it,j},
    with
    .. math::
        \eta_{it,j}\stackrel{\text{i.i.d}}{\sim} N(0,1) \qquad j\in \{1,2\}.
    The regressors are correlated with :math:`\lambda_i`, :math:`f_t`, and the product
    :math:`\lambda_i' F_t`.
    The regression error
    .. math::
        \epsilon_{it} \stackrel{\text{i.i.d}}{\sim}N(0,4).
    """
    # Set parameters
    p = 3
    gamma = 0
    delta = 0
    factor = np.random.normal(loc=0, scale=1, size=(2, T))
    lambda_ = np.random.normal(loc=0, scale=1, size=(2, N))
    X, Y = _dgp_fixed_effect_panel_data(
        T, N, beta1, beta2, mu, gamma, delta, factor, lambda_
    )
    X = X[0:p, :, :]
    return X, Y


def dgp_interactive_fixed_effects_model_with_common_and_time_invariant(
    T, N, *, beta1, beta2, mu, gamma, delta
):
    r"""
    Monte carlo data generate processor for "Interactive Fixed Effects Model with Common
    Regressors and Time-invariant Regressors"

    Parameters
    ----------
    T : int
        Sample size of time
    N : int
        Sample size of entity
    beta1 : float
        Coefficient of x1
    beta2 : float
        Coefficient of x2
    mu : float
        Constant
    gamma : float
        Coefficient of x_i
    delta : float
        Coefficient of w_t

    Returns
    -------
    X : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y : array-like
        Simulate data of Dependent (left-hand-side) variable (time by entity).

    Notes
    -----
    .. math::
        y_{it} = \beta_{1}x_{it,1}+\beta_{2}x_{it,2}+ \mu+ x_{i}\gamma +w_{t}\delta +
        \lambda_{i}'F_{t}+\epsilon_{it}
    where
    .. math::
    \[
        \lambda_i = \binom{\lambda_{i1}}{\lambda_{i2}} \stackrel{\text{i.i.d}}{\sim}
        N(0,I_2),
    \]
    \[
        F_t =\binom{F_{t1}}{F_{t2}}\stackrel{\text{i.i.d}}{\sim}N(0,I_2).
    \]
    The regressors are generated according to:
    .. math::
        X_{it,j}= 1+\lambda_{i1}F_{t1}+\lambda_{i2}F_{t2}+\lambda_{i1}+\lambda_{i2}+
        F_{t1}+F_{t2}+\eta_{it,j},
    with
    .. math::
        \eta_{it,j}\stackrel{\text{i.i.d}}{\sim} N(0,1) \qquad j\in \{1,2\}.
    Additionally,  we set
    .. math::
        x_{i}=\lambda_{i1}+\lambda_{i2}+e_{i},
        \qquad e_{i} \stackrel{\text{i.i.d}}{\sim} N(0,1)
    and
    .. math::
        w_{t}=F_{t1}+F_{t2}+\eta_{t},
        \qquad \eta_{t} \stackrel{\text{i.i.d}}{\sim} N(0,1),
    so that :math:`x_{i}` is correlated with :math:`\lambda_i` and :math:`w_t` is
    correlated with :math:`f_t`.
    """
    # Set parameters
    factor = np.random.normal(loc=0, scale=1, size=(2, T))
    lambda_ = np.random.normal(loc=0, scale=1, size=(2, N))
    X, Y = _dgp_fixed_effect_panel_data(
        T, N, beta1, beta2, mu, gamma, delta, factor, lambda_
    )
    return X, Y


def _dgp_fixed_effect_panel_data(T, N, beta1, beta2, mu, gamma, delta, factor, lambda_):
    # Set parameters
    iota = np.array([[1], [1]])
    mu1 = mu2 = c1 = c2 = 1
    # Generate variables
    eta_1 = np.random.normal(loc=0, scale=1, size=(N, T))
    eta_2 = np.random.normal(loc=0, scale=1, size=(N, T))
    eps = np.random.normal(loc=0, scale=2, size=(N, T))
    e = np.random.normal(loc=0, scale=1, size=(N, 1))
    eta = np.random.normal(loc=0, scale=1, size=(1, T))
    # Calculate intermediate variables
    iota_lambda = lambda_.T.dot(iota)
    iota_factor = iota.T.dot(factor)
    lambda_factor = lambda_.T.dot(factor)
    x = iota_lambda + e
    w = iota_factor + eta
    # Simulate data
    X_1 = (
        mu1
        + c1 * lambda_factor
        + np.tile(iota_lambda, (1, T))
        + np.tile(iota_factor, (N, 1))
        + eta_1
    )
    X_2 = (
        mu2
        + c2 * lambda_factor
        + np.tile(iota_lambda, (1, T))
        + np.tile(iota_factor, (N, 1))
        + eta_2
    )
    X_3 = np.ones(shape=(N, T))
    X_4 = np.tile(x, (1, T))
    X_5 = np.tile(w, (N, 1))
    X = np.stack([X_1, X_2, X_3, X_4, X_5])
    Y = (
        beta1 * X_1
        + beta2 * X_2
        + mu * X_3
        + gamma * X_4
        + delta * X_5
        + lambda_factor
        + eps
    )
    return X, Y
