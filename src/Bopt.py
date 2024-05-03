""" Basic Bayesian Optimizaiton : Designed for PyCall.jl

    Hard-coded for arbitrary Parameters and noise-free scalar objective. 
    
    Using Expected Improvement and 5/2-Matern Kernel. 
    
    For loss, we are maximizing the loss function. 
        This is a consequence of the fact the Bayesian Optimization sign convention. 
"""
import os, sys
import os.path
import math
import numpy as np
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from warnings import catch_warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings(action='once')
import pandas as pd


def Restart(interval):
    """ Restart Bayesian Optimization from the same directory if possible. 

    Args:
        interval (dictionary): Set of intervals or bounds for all parameters. 

    Returns:
        x_set   : Set of free parameters corresponding to `interval`.
        y_set   : Set of evaluated Objective function (negatie of loss fucntion).
        idx_set : Set of indices corresponding to x and y_set
    """
    if os.path.isfile("Bopt_Log.csv"): 
        df_read = pd.read_csv("Bopt_Log.csv", sep="\t", index_col=0)
        D = df_read.to_numpy()
        idx_set = np.vstack([i for i in D[:,0]])
        y_set = np.vstack([i for i in D[:,-1]])
        x_set = D[:,1:-1]
    else:
         x_set = np.array([])  
         y_set = np.array([])  
         idx_set = np.array([[0.0]])  
    return x_set, y_set, idx_set



def surrogate(model, X) : 
    """Helper Function to supress warning when evaluating surrogate models. 

    Args:
        model (GP model): Trained surrogoate GP model 
        X (array): Parameters to evaluate. 

    Returns:
        model predictions (float) : Model predictions of the GP. 
    """
    warnings.filterwarnings("ignore")
    return model.predict(X, return_std=True)



def Expected_Improvement(X, XS, model, explore) :
    """Expected Improvement Surrogate function

    Args:
        X (array): All evaluated parameters. 
        XS (array): Trial parameter for EI calculation. 
        model (GP): Gaussian Process model fit to data
        explore (float): Degree of exploration.  

    Returns:
        EI (float): Expected Improvmeent score. 
    """
    # Find best Score of Data-points. 
    yhat, _ = surrogate(model, X) 
    best = max(yhat) # Current best of the surrogate model.
    
    # Calculate Mean and StdDev via Surrogate Model. 
    mu, std =  surrogate(model, XS) 
    
    # Expected Improvment Calculation.
    I = mu - best - explore 
    Z = np.divide(I, std+1e-8)
    EI = I*scipy.stats.norm.cdf(Z) + std*scipy.stats.norm.pdf(Z)
    
    return EI 



def Opt_Acquisition(X, model, bounds, explore=0.0): 
    """Optimize the Acquisition function to find the next set of parameters to evaluate. 

    Args:
        X (array): Parameters that have been already evaluated. 
        model (GP): Gaussian Process as the surrogate model.
        bounds (dictionary): Set of bounds for all parameters. 
        explore (float, optional): Exploration rate. Defaults to 0.0.

    Returns:
        (array): Set of parameters to evaluate for next B.O. iteration. 
    """
    to_search = np.array([np.random.uniform(low=bounds[b][0], high=bounds[b][1], size=50-len(bounds)) for b in bounds])
    XR = to_search.T
    
    # Calculates optimum EI score
    EIScores = Expected_Improvement(X, XR, model, explore=explore)
    return np.array([XR[np.argmax(EIScores)]])
        