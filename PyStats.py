import numpy as np
from numpy.linalg import eig
from numpy import mean
from scipy import stats
from math import factorial

def variance(x):
    mean = np.mean(x)
    total = 0
    for element in x:
        total += (element - mean)**2
    return total/(len(x)-1)

def std(x):
    return np.sqrt(variance(x))

def skew(x):
    var = variance(x)
    mean = np.mean(x)
    total = 0
    for element in x:
        total += (element - mean)**3
    return total / (len(x) * (var**(3/2)))

def covariance(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    total = 0
    for i in range(len(x)):
        total += (x[i] - mean_x)*(y[i] - mean_y)
    
    return total/(len(x)-1)

def spearman(x, y):
    """Calculates Spearman Rank Coefficient for two arrays of data,
    x and y."""
    sorted_x_idx = stats.rankdata(x)
    sorted_y_idx = stats.rankdata(y)
    total = 0
    N = len(x)
    for i in range(N):
        total += (sorted_x_idx[i] - sorted_y_idx[i])**2
        
    return (1 - (6*total) / (N * (N**2 - 1)))

def binomial(r, p, n):
    frac = (factorial(n) / (factorial(r)*factorial(n-r)))
    return p**r * (1 - p)**(n - r) * frac

def poisson(r, lam):
    return (lam**r * np.exp(-lam)) / factorial(r)

def poisson_cumulative(r_range, lam, limits=True):
    """Calculates the cumulative probability for a Poisson distribution.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    r_range: array-like
              if limits=True, then this array should contain the upper and lower limit
              of the r values that are desired to be calculated. The r value is the number
              of outcomes.
              Otherwise, this should contain the array of r_values that are desired to be
              summed.
    
    lam: integer
         the expected number of outcomes.
    
    limits: boolean
            default is True.
            True - r_range will be treated as an array containing the lower and upper limits
            respectively of the range of r values to be calculated.
            False - r_range will be treated as an array containing the r values to be
            calculated.
            
    """
    total = 0
    if limits == True:
        if len(r_range) != 2:
            print("When limits=True, r_range should contain simply a lower an upper limit")
            return None
            
        a = r_range[0]
        b = r_range[1]
        r_values = np.linspace(a, b, 1 + (b - a))

        for r in r_values:
            total += poisson(r, lam)
            
    else:
        for r in r_range:
            total += poisson(r, lam)
        
    return total

def gaussian(x, mu, sigma):
    return (1 / (sigma*np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2)/(2*sigma**2))

def Gamma(n):
    return factorial(n-1)

def chi_squared(chisq, nu):
    return (2**(-nu/2) * chisq**(nu/2 - 1) * np.exp(-chisq/2)) / Gamma(nu/2)

def uniform(x, N):
    return 1/N

def weighted_av_single(x, errors):
    """Calculates the weighted average of a set of measurements for a 
    single observation.
    
    Returns the weighted average and the error.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    x: list containing the measurements
    errors: list containing the errors.
    """
    
    total_top = 0
    total_bottom = 0
    
    for i in range(len(x)):
        var = errors[i]**2
        total_top += x[i] / var
        total_bottom += 1/var
        
    mean_x = total_top/total_bottom
    sigma_x = 1/np.sqrt(total_bottom)
        
    return mean_x, sigma_x


def weighted_av_set(measurements, errors, correlation, precision=None):
    """Calculates the weighted average of a set of measurements for a 
    set of observables.
    
    Returns the weighted average and the error.
    
    CURRENTLY ONLY BUILT FOR 2D.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    measurements: list of lists containing the measurements for each
                    experiment
    errors: list of lists containing the errors for each experiment
    correlation: list of lists containing the correlations for each 
                    experiment
    """
    V_list = []
    x_list = []
    size = len(measurements)
    
    for i in range(size):
        corr_mat = np.ones((size, size))
        cov_mat = np.ones((size, size))*correlation[i][0]*errors[i][0]*errors[i][1]
        for j in range(len(errors[i])):
            cov_mat[j, j] = errors[i][j]**2
            
            #corr_mat[j, -j-1] = correlation[i][0]

        inv_cov = np.linalg.inv(cov_mat)
        
        V_list.append(inv_cov)
        x_list.append(inv_cov @ measurements[i])
            
    V = np.linalg.inv(sum(V_list))
    x = V @ (sum(x_list))
    
    if precision == None:
        return x, V
    
    else:
        return np.round(x, precision), np.round(V, precision)
            
            

def hypothesis_ratio(data, hypotheses=[gaussian, uniform], params=[[0, 1], [10]]):
    """Calculates the product of all ratios between the two hypotheses
    for a given set of data.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    data: the data to which the two hypotheses apply
    hypotheses: the two distributions that we wish to compare for the data
    params: the parameters for the given hypotheses"""
    h0, h1 = hypotheses
    PI = 1
    for w in data:
        PI *= h0(w, *params[0]) / h1(w, *params[1])
    
    return PI

def bayes_theorem(p_c_a, p_c_b, p_a):
    """
    Calculates Bayes' Theorem for a scenario with two possible main
    outcomes.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    p_c_a: probability of event c given a is true
    p_c_b: probability of event c given b is true
    p_a: probability of event a being true
    
    -+-+-+-+-+-
    EXAMPLE
    -+-+-+-+-+-
    
    Consider a test that correctly predicts infection at a rate of 98%,
    and incorrectly predicts infection at a rate of 0.05%. The percentage
    of infected individuals in the population is 0.01%. The probability
    that someone is infected given a positive test result can be found
    using the following logic:
    
    - Event a is infection
    - Event b is health
    - Event c is a positive test
    
    bayes_theorem(0.98, 0.0005, 0.0001)
    
    """
    
    p_b = 1 - p_a
    p_c = p_c_a*p_a + p_c_b*p_b
    
    p_a_c = (p_c_a*p_a) / (p_c)
    
    return p_a_c



def chisq_scan(measurements, errors, scan_range, increment=.1,
              same_errors=False):
    """Estimates the parameters that minimize chi-squared using
    scanning across a desired range.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    measurements: array
                  list of measurements
    
    errors: array
            list of errors. If all errors are equal, can simply use
            a list of length 1 with the error, and set same_errors=True.
            
    scan_range: tuple
                the range across which you want parameters to be searched.
                
    increment: float 
               the increments between consecutive searches.
               
    same_errors: boolean
                 default is False. Set to true if all errors are equal,
                 and allow errors to only contain one element with the
                 error.
                 
    -+-+-+-+-+-
    RETURNS
    -+-+-+-+-+-
    
    x_min: the bin that corresponds to the lowest total chi-squared value
    
    sigma: the error associated to x_min
    
    chi_totals: the chi-squared total values for each bin
                 
    """
    a, b = scan_range
    x_values = np.arange(a, b+increment, increment)
    N = len(measurements)
    
    chi_totals = []
    
    if same_errors == True:
        errors = [errors[0] for i in range(N)]
    
    for x in x_values:
        chi_values = []
        for i in range(len(measurements)):
            chi = ((measurements[i] - x) / errors[i])**2
            chi_values.append(chi)
        chi_totals.append(sum(chi_values))
        
    idx_min = np.argmin(chi_totals)
    delta_chi = 0
    idx_low = idx_min
    while delta_chi < 1:
        idx_low -= 1
        delta_chi = chi_totals[idx_low] - chi_totals[idx_min]

    delta_chi = 0
    idx_high = idx_min
    while delta_chi < 1:
        idx_high += 1
        delta_chi = chi_totals[idx_high] - chi_totals[idx_min]
    
    x_min = x_values[idx_min]
    chi_errors = []
    for i in range(-1, 2, 2):
        delta_chi = 0
        idx = idx_min
        while delta_chi < 1:
            idx += i
            delta_chi = chi_totals[idx] - chi_totals[idx_min]
        chi_errors.append(abs(x_min - x_values[idx]))
    
    sigma = np.mean(chi_errors)
    return x_min, sigma, chi_totals



def lin_least_squares_2d(x, y, sig):
    """
    Estimates the parameters a and b for a 2d dataset for a model following
    y = ax + b.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    x: array-like
        contains x values
        
    y: array-like
        contains y values
        
    sig: float
         error associated with the values
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x2_mean = np.mean([i**2 for i in x])
    xy_mean = np.mean([i*j for i, j in zip(x, y)])
    
    N = len(x)
    
    def a_func(xy_mean, x_mean, y_mean, x2_mean):
        return (xy_mean - x_mean*y_mean)/(x2_mean - x_mean**2)
    
    def b_func(x_mean, y_mean, a):
        return y_mean - a*x_mean

    def sig_a_func(sig, N, x_mean, x2_mean):
        return np.sqrt((sig**2)/(N*(x2_mean - x_mean**2)))
    
    def sig_b_func(sig, N, x_mean, x2_mean):
        return np.sqrt((sig**2 * x2_mean)/(N*(x2_mean - x_mean**2)))
    
    a = a_func(xy_mean, x_mean, y_mean, x2_mean)
    b = b_func(x_mean, y_mean, a)
    sig_a = sig_a_func(sig, N, x_mean, x2_mean)
    sig_b = sig_b_func(sig, N, x_mean, x2_mean)
    
    return a, b, sig_a, sig_b


def fisher_disc(mu_a, mu_b, V_a, V_b):
    """Calculates the fisher discriminant coefficients for a dataset
    in which we know which elements are of class A and which are of class
    B, using the mean and variance of each class.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    mu_a / mu_b: matrix with the mean for class A / B
    
    V_a / V_b: matrix with variance for class A / B
    """
    delta_mu = mu_a - mu_b
    W = V_a + V_b
    W_inv = np.linalg.inv(W)
    return W_inv @ delta_mu

def bayes_classifier(xs, ys=None, distributions_x=[gaussian, uniform],
                        distributions_y=None, params_x=[[0, 1], [10]],
                        params_y=None, categories=None):
    
    """Creates a Bayesian classifier to categorise events in a dataset.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    xs: array-like. 
        the x data that we want to classify, or simply the data for a 
        one dimensional dataset.
        
    ys: array-like.
        the y data for a 2d dataset.
    
    distributions_x: list.
                   contains the functions that act as the distributions
                   for each classifier model for the x data (or simply
                   the data if ys=None).
                   
    distributions_y: list.
                    contains the functions that act as the distributions
                    for each classifier model for the y data.
                   
    params_x / params_y: list of lists.
            containing the parameters for the distribution functions for
            the x / y data
            
    categories: default=None.
                otherwise should be a list containing the class names
                as strings
                
    -+-+-+-+-+-
    RETURNS
    -+-+-+-+-+-
    class_list: list containing:
                - indices of the classified events corresponding to
                the distribution in the distributions list
                - names of the classified class if categories != None.
    """
    
    class_list = []
    p_list = np.zeros((len(xs), len(distributions_x)))
    if ys == None:
        for row, x in enumerate(xs):
            for i in range(len(distributions_x)):
                p_list[row, i] = (distributions_x[i](x, *params_x[i]))

    
    else:
        for row, (x, y) in enumerate(zip(xs, ys)):

            for i in range(len(distributions_x)):
                p_a_x = distributions_x[i](x, *params_x[i])
                p_a_y = distributions_y[i](y, *params_y[i])
                p_list[row, i] = (p_a_x*p_a_y)
            
    for data_point in p_list:    
        classification = np.argmax(data_point)
        if categories != None:
            classification = categories[classification]
        class_list.append(classification)
        
    return class_list



def separation(mu_a, mu_b, sig_a, sig_b):
    """
    Calculates the separation between two distributions of events.
    
    -+-+-+-+-+-
    PARAMETERS
    -+-+-+-+-+-
    
    mu_a / mu_b: mean of the distribution a / b
    
    sig_a / sig_b: error of distribution a / b
    """
    
    return (mu_a - mu_b)**2 / (sig_a**2 + sig_b**2)