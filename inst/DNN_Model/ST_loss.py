import torch
from torch.distributions.studentT import StudentT
from scipy import integrate
from MISC import torch_to_numpy
import numpy as np

def dTPSC(x,w,theta,sigma,delta):
    """
    The probability density function of the TPSC-Student-t distribution

    Parameters
    ----------
    x : torch.Tensor
        the input from the density function of the TPSC-Student-t distribution.
    w : torch.Tensor
        the skewness parameter.
    theta : torch.Tensor
        the location parameter.
    sigma : torch.Tensor
        the standard deviation parameter.
    delta : torch.Tensor
        the degree of freedom parameter.

    Returns
    -------
    output : torch.Tensor
        the calculated density of the TPSC-Student-t distribution.
    """
    sigma1 = sigma * torch.sqrt(w/(1.0 - w))
    sigma2 = sigma * torch.sqrt((1.0 - w)/w)
    # define the t distribution class
    dist = StudentT(df = delta,loc=0.0, scale=1.0)
    p1 = w*2.0/sigma1*torch.exp(dist.log_prob((x-theta)/sigma1))*(x<theta + 0.0)
    p2 = (1.0-w)*2.0/sigma2*torch.exp(dist.log_prob((x-theta)/sigma2))*(x>=theta + 0.0)
    output = p1 + p2
    return output

@np.vectorize
def pTPSC(x,w,theta,sigma,delta):
    """
    The cumulative density function of the TPSC-Student-t distribution

    Parameters
    ----------
    x : torch.Tensor
        the input from the density function of the TPSC-Student-t distribution.
    w : torch.Tensor
        the skewness parameter.
    theta : torch.Tensor
        the location parameter.
    sigma : torch.Tensor
        the standard deviation parameter.
    delta : torch.Tensor
        the degree of freedom parameter.

    Returns
    -------
    output : float
        the calculated cumulative density of the TPSC-Student-t distribution.
    """
    try:
        x,w,theta,sigma,delta = torch_to_numpy(x,w,theta,sigma,delta)
    except:
        x = torch.tensor(x)
        w = torch.tensor(w)
        theta = torch.tensor(theta)
        sigma = torch.tensor(sigma)
        delta = torch.tensor(delta)
    output = integrate.quad(lambda x: dTPSC(x = x,
                                            w = w,
                                            theta = theta,
                                            sigma = sigma,
                                            delta = delta).detach().cpu().numpy(), 
                            -np.inf, 
                            x)[0]
        
    return output

def loss(x,w,theta,sigma,delta):
    """
    The negative log density function of the TPSC-Student-t distribution

    Parameters
    ----------
    x : torch.Tensor
        the input from the density function of the TPSC-Student-t distribution.
    w : torch.Tensor
        the skewness parameter.
    theta : torch.Tensor
        the location parameter.
    sigma : torch.Tensor
        the standard deviation parameter.
    delta : torch.Tensor
        the degree of freedom parameter.

    Returns
    -------
    output : torch.Tensor
        the calculated negative log density function of the TPSC-Student-t distribution.
    """
    output = dTPSC(x = x,
                   w = w,
                   theta = theta,
                   sigma = sigma,
                   delta = delta)
    output = -1.0 * torch.log(output)
    return output
