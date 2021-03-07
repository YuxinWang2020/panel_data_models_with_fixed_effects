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
    X_list : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y_list : array-like
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
    pass


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
    X_list : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y_list : array-like
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
    pass


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
    X_list : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y_list : array-like
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
    pass


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
    X_list : array-like
        Simulate data of Exogenous or right-hand-side variables (variable by time by
        entity).
    Y_list : array-like
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
    pass
