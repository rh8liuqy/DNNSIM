import numpy as np
import torch
from scipy import stats
from torch.distributions.binomial import Binomial

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = "cuda"
else :
    device = "cpu"

def np_narray_convert(*args):
    """
    Conver input variables as a list of numpy.ndarray.

    Parameters
    ----------
    *args : 
        A collection of input variables.

    Returns
    -------
    output : list
        A list of numpy.ndarray.

    """
    output = [None] * len(args)
    for i in range(len(args)):
        if type(args[i]) == np.ndarray:
            output[i] = args[i]
        else :
            output[i] = np.array(args[i])
    return output

def torch_to_numpy(*args):
    """
    Conver input variables as a list of numpy.ndarray.

    Parameters
    ----------
    *args : 
        A collection of input variables.

    Returns
    -------
    output : list
        A list of numpy.ndarray.

    """
    output = [None] * len(args)
    for i in range(len(args)):
        if type(args[i]) == np.ndarray:
            output[i] = args[i]
        else :
            output[i] = args[i].detach().cpu().numpy()
    return output

def numpy_to_torch(*args):
    """
    Conver input variables as a list of numpy.ndarray.

    Parameters
    ----------
    *args : 
        A collection of input variables.

    Returns
    -------
    output : list
        A list of torch.Tensor.

    """
    output = [None] * len(args)
    for i in range(len(args)):
        if type(args[i]) == torch.Tensor:
            output[i] = args[i]
        else :
            output[i] = torch.from_numpy(args[i]).to(device, dtype = torch.float32)
    return output

def DNN_prediction_DNN1(model,beta_estimation,start,stop,num):
    """
    The function to estimate the single index function using a trained DNN model.
    
    Parameters
    ----------
    model : DNN.DNN1
        The defined DNN model.
    beta_estimation : torch.Tensor
        The point estimation of beta.
    start : float
        The starting value of the eta vector.
    stop : float
        The ending value of the eta vector.
    num : int
        Number of etas to generate..

    Returns
    -------
    output : dict
        A dictionary consisting of eta vector and associated predicted value.

    """
    p = model.p
    X = np.zeros(shape = (num,p))
    if type(beta_estimation) == torch.Tensor :
        beta_estimation = beta_estimation.cpu().detach().numpy()
    beta0 = beta_estimation[0]
    X[:,0] = (X[:,0] + 1.0) / beta0
    X[:,0] = X[:,0] * np.linspace(start = start, stop = stop, num = num)
    X = torch.from_numpy(X)
    X = X.to(device, dtype = torch.float32)
    model_output = model(X)
    SIM_output = model_output[4]
    output = {"eta": np.linspace(start = start, stop = stop, num = num),
              "predicted_value": SIM_output.cpu().detach().numpy()}
    return output

def bootstrap_index(n,seed):
    n = int(n)
    seed = int(seed)
    # define random number generator
    rng = np.random.default_rng(seed)
    # define the original seed
    index = np.array(np.arange(0,n,1))
    # sample with replacement
    output = rng.choice(a = index, size = (n,),replace = True)
    return output

# the random number generator for generative bootstrap sampler
def rW(n,seed):
    n = int(n)
    seed = int(seed)
    # define random number generator
    rng = np.random.default_rng(seed)
    X = rng.exponential(scale = 1.0, size = (n,))
    X_sum = np.sum(X)
    output = X / X_sum
    return output

def Goodness_of_Fit_Test(residuals,w,sigma,delta):
    """
    The goodness of fit test associated with the DNNSIM model

    Parameters
    ----------
    residuals : numpy.ndarray
        The residuals from the fitted DNNSIM model.
    w : numpy.ndarray
        The estimated skewness parameter.
    sigma : numpy.ndarray
        The estimated standard deviation parameter.
    delta : numpy.ndarray
        The estimated degree of freedom parameter.

    Returns
    -------
    output : KstestResult
        Results related to the Kolmogorov-Smirnov test for goodness of fit.

    """
    # import rTPSC inside the function to avoid the circular import issue.
    from ST_RNG import rTPSC
    output = stats.kstest(residuals,
                          rTPSC(n = int(2e5), 
                                w = w, 
                                theta = 0.0, 
                                sigma = sigma, 
                                delta = delta, 
                                seed = 100))
    return output

def bernstein_matrix(L, x):
    """
    The function to create a matrix consisting of Bernstein basis polynomials.

    Parameters
    ----------
    L : int
        the degree of Bernstein basis polynomials.
    x : ndarray,shape(k,)
        The input X from f(X). The function f(X) is the one that is approximated by Bernstein basis polynomials.

    Returns
    -------
    B : ndarray, shape(k,L+1)
        A matrix consisting of Bernstein basis polynomials.

    """
    L = int(L)
    if type(x) == np.ndarray:
        x = torch.from_numpy(x).to(device = device, dtype = torch.float32)
    
    # determine the size of x
    num_points = x.shape[0]
    
    # calculate the range of x
    a = torch.min(x)
    b = torch.max(x)
    
    # Initialize the matrix with shape (num_points, L + 1)
    B = torch.zeros((num_points, L + 1)).to(device = device, dtype = torch.float32)
    
    # ensure x_trans ranges from 0 to 1
    x_trans = (x - a)/(b - a)
    
    # Fill in the matrix with Bernstein basis polynomials
    for i in range(L + 1):
        B[:,i] = torch.exp(Binomial(total_count=L,probs=x_trans).log_prob(torch.tensor(data = float(i)).to(device = device, dtype = torch.float32)))
    
    # define the output
    output = B
    
    return output