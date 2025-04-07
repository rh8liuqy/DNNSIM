import numpy as np
from MISC import np_narray_convert

def rTPSC(n, w, theta, sigma, delta, seed):
    """
    Generate random quantities from the TPSC-Student-t distribution.

    Parameters
    ----------
    n : int
        The sample size.
    w : float
        The weight parameter. 0 <= w <= 1.
    theta : float or numpy.ndarray
        The location parameter.
    sigma : float
        The scale parameter. It must be positive.
    delta : float
        The degree of freedom. It must be positive.
    seed : int
        The random seed.

    Returns
    -------
    output : numpy.ndarray
        The simulated data from a TPSC-Student-t distribution.

    """
    # ensure seed and n are integers
    seed = int(seed)
    n = int(n)
    # convert other inputs as np.narray
    w, theta, sigma, delta = np_narray_convert(w, theta, sigma, delta)
    # ensure that theta is not a matrix
    if len(theta.shape) == 2:
        theta = theta[:,0]
    # creation of the random generator
    rng = np.random.default_rng(seed)
    # generate random quantitiles 
    X1 = rng.standard_t(df = delta, size = (n,))
    X2 = rng.standard_t(df = delta, size = (n,))
    # the left skewed component
    X1 = -np.abs(X1) * sigma * np.sqrt(w/(1.0-w))
    # the right skewed component
    X2 = np.abs(X2) * sigma * np.sqrt((1.0-w)/w)
    # the latent variabel Z
    Z = rng.binomial(n = 1, p = w, size = (n,))
    # define the output
    output = X1 * Z + X2 * (np.ones((n,)) - Z) + theta
    return output

def reg_simu(n, beta, w, sigma, delta, seed):
    """
    Generate data from the single-index model with the noise from the TPSC-Student-t distribution.

    Parameters
    ----------
    n : int
        The sample size.
    beta : np.narray
        The coefficients of the covariates.
    w : float
        The weight parameter. 0 <= w <= 1.
    sigma : float
        The scale parameter. It must be positive.
    delta : float
        The degree of freedom. It must be positive.
    seed : int
        The random seed.

    Returns
    -------
    output : dict
        A dictionary consisting of the response variable y and the covariates matrix X.

    """
    # ensure seed and n are integers
    seed = int(seed)
    n = int(n)
    # convert other inputs as np.narray
    beta, w, sigma, delta = np_narray_convert(beta, w, sigma, delta)
    # ensure that beta is not a matrix
    if len(beta.shape) == 2:
        beta = beta[:,0]
    # ensure the norm of beta is 1
    beta = beta / np.linalg.norm(beta)
    # the length of beta
    p = beta.shape[0]
    # creation of the random generator
    rng = np.random.default_rng(seed)
    # simulate the design matrix
    X = rng.uniform(low = -3.0, high = 3.0, size = (n,p))
    eta = X @ beta
    # calculate the location parameter
    theta = 1.0 / (1.0 + np.exp(-eta))
    # generate the response variable
    y = rTPSC(n = n, w = w, theta = theta, sigma = sigma, delta = delta, seed = seed)
    # define the output
    output = {"y": y,
              "X": X}
    return output
