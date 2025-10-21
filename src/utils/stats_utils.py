import pandas as pd
import numpy as np
import scipy.stats as stats
# from scipy.interpolate import interp1d

#* --------------------------------------------------------------------------------
#* Essential imports for GMM fitting with SciPy and for plotting
#* --------------------------------------------------------------------------------
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt




def shuffle(x,y):
    """Random shuffle two numpy arrays
    Arguments:
        x, y: two numpy arrays (not necessarily of same length)
    Returns: two shuffled numpy arrays with same respective lengths
    
    Requirements: import numpy as np
    
    Author: Gary Mamon (gam AAT iap.fr)
    """

    xy = np.concatenate((x,y))
    np.random.shuffle(xy)
    return xy[0:len(x)],xy[len(x):]

def CompareByShuffles(x,y,N=10000,stat='median',val=1,verbosity=0):
    """Compare two distributions 
        according to the difference in a statsitic
        using random shuffles
        
       Arguments :
           x, y: two numpy arrays (not necessarily of same length)
           N: number of random shuffles (default: 100000)
           stat: statistic used for difference (default: 'median')
           val: optional parameter for count of x and y above value
           
       Returns: f
           fraction of trials with greater stat_sample1 - stat_sample2
           
       Requires: 
           import numpy as np
           import scipy.stats as stats
           
       Author: Gary Mamon (gam AAT iap.fr)
        """
    # statistic on data
    if stat == 'median':
        stat_data = np.median(x) - np.median(y)
    elif stat == 'mean':
        stat_data = np.mean(x) - np.mean(y)
    elif stat == 'stdev':
        stat_data = np.std(x) - np.std(y)
    elif stat == 'kurt':
        stat_data = stats.kurtosis(x) - stats.kurtosis(y)
    elif stat == 'count':
        stat_data = len(x[x>val])/len(x) - len(y[y>val])/len(y)
    else:
        stat_data = stat(x) - stat(y)
    if verbosity > 0:
        print("stat_data=",stat_data)
        
    # loop over trials
    stat_shuf = np.zeros(N)
    for n in range(N):
        # shuffle data into same size components
        xshuffle, yshuffle = shuffle(x,y)
        
        # statistic on shuffled data
        if stat == 'median':
            stat_shuf[n] = np.median(xshuffle) - np.median(yshuffle)
        elif stat == 'mean':
            stat_shuf[n] = np.mean(xshuffle) - np.mean(yshuffle)
        elif stat == 'stdev':
            stat_shuf[n] = np.std(xshuffle) - np.std(yshuffle)
        elif stat == 'kurt':
            stat_shuf[n] = stats.kurtosis(xshuffle) - stats.kurtosis(yshuffle)
        else:
            stat_shuf[n] = stat(xshuffle) - stat(yshuffle)
            
    # check fraction of data with higher statistic
    Nworse = np.sum(stat_shuf > stat_data)
    if verbosity > 0:
        print("median stat of shuffled = ",np.median(stat_shuf))
    
    return Nworse/N


def CompareByShuffles_normalized(x,y,N=10000,stat='median',val=1,verbosity=0):
    """Compare two distributions 
        according to the difference in a statsitic
        using random shuffles
        but returns 1-f if f>0.5
        
       Arguments :
           x, y: two numpy arrays (not necessarily of same length)
           N: number of random shuffles (default: 100000)
           stat: statistic used for difference (default: 'median')
           val: optional parameter for count of x and y above value
           
       Returns: f
           fraction of trials with greater stat_sample1 - stat_sample2
           
       Requires: 
           import numpy as np
           import scipy.stats as stats
    """
    shuf = CompareByShuffles(x,y,N,stat,val,verbosity)
    if (shuf<=0.5):
        return shuf
    else:
        return 1-shuf

def median_errors(series):
    """compute the error on the median of a series

    Args:
        series (pandas.Series): series of values
    """
    N = series.shape[0]
    return((series.quantile(q=0.84)-series.quantile(q=0.16))/(2*np.sqrt(2*N/np.pi)))

def median_with_errors_str(series,decimals=3):
    """Compute the error on the median of a series and return a string like "3 +/- 0.1"

    Args:
        series (pandas.Series): series of values
        decimals (int, optional): number of decimals. Defaults to 3.
    """
    N = series.shape[0]
    return(f'{series.median():.{decimals}f} +/- {(series.quantile(q=0.84)-series.quantile(q=0.16))/(2*np.sqrt(2*N/np.pi)):.{decimals}f}')
    



def bootstrap_std(sample,n_samples=10000):
    """Compute the standard deviation of a sample using bootstrap resampling

    Args:
        sample (pandas.DataFrame): sample of values
        n_samples (int, optional): number of time to perform sampling. Defaults to 10000.

    Returns:
        pandas.Series: standard deviation of the sample
    """
    # Create an empty list to store the bootstrap estimates
    bootstrap_estimates = []

    # Perform the bootstrap resampling
    for i in range(n_samples):
        # Sample data with replacement
        resample = sample.sample(frac=1, replace=True)
        # Calculate the standard deviation of the resample
        std_resample = resample['values'].std()
        # Add the estimate to the list
        bootstrap_estimates.append(std_resample)

    # Calculate the standard deviation of the bootstrap estimates
    std = pd.Series(bootstrap_estimates).std()
    
    return std


    def V_disp_gapper(gals):
        """Computes the gapper velocity dispersion of a group
        According to Wainer & Thissen (1976)

        Args:
            gals (pandas.DatqaFrame): DataFrame of galaxies with 'z' column

        Returns:
            real: velocity dispersion
        """
        vd = gals['z'].sort_values().reset_index(drop=True)
        v= vd.values
        n = len(v)
        w = np.arange(1, n) * np.arange(n-1, 0, -1)
    
        g = np.diff(v)
        sigma_z = (np.sqrt(np.pi))/(n*(n-1)) * np.dot(w, g)
        
        z_group = gals['z'].mean()
        Vdisp = c*sigma_z/(1+z_group)

        return Vdisp



#* --------------------------------------------------------------------------------
#* Define GMM-related functions: gmm_pdf, estimate_kl_divergence, parameter conversions,
#* objective function and multi-initialization GMM fitting.
#* --------------------------------------------------------------------------------

def gmm_pdf(X, means, covs, weights):
    """
    Compute the probability density function of a 2-component Gaussian Mixture Model (GMM) 
    at points X. Each GMM component has its own mean, covariance, and weight.
    """
    n_samples = X.shape[0]
    pdf_values = np.zeros(n_samples)
    for i in range(2):
        pdf_values += weights[i] * multivariate_normal.pdf(
            X, mean=means[i], cov=covs[i], allow_singular=True
        )
    return pdf_values

def estimate_kl_divergence(X, means, covs, weights, n_bins=50):
    """
    Estimate KL divergence between empirical data and a GMM using histogram approximation.
    We create a 2D histogram of the data, evaluate the GMM on the histogram grid, 
    and sum p_data * log(p_data / p_model).
    """
    #* Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(X[:, 0], X[:, 1], bins=n_bins, density=True)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
    
    gmm_values = gmm_pdf(grid_points, means, covs, weights).reshape(X_grid.shape)
    
    epsilon = 1e-10
    hist = hist + epsilon
    gmm_values = gmm_values + epsilon
    kl_div = np.sum(hist * np.log(hist / gmm_values.T))
    
    return kl_div

def params_to_gmm(params, constrain_means=False):
    """
    Convert a flat parameter vector into GMM parameters (means, covariances, weights). 
    Optionally constrain the second mean's y-coordinate to [-12, -10].
    """
    mean1 = params[0:2]
    mean2 = params[2:4]
    if constrain_means:
        mean2[1] = np.clip(mean2[1], -12.0, -10.0)
    
    #* Build covariance matrices from a Cholesky-like representation
    L1 = np.zeros((2, 2))
    L1[0, 0] = np.exp(params[4])
    L1[1, 0] = params[5]
    L1[1, 1] = np.exp(params[6])
    
    L2 = np.zeros((2, 2))
    L2[0, 0] = np.exp(params[7])
    L2[1, 0] = params[8]
    L2[1, 1] = np.exp(params[9])
    
    cov1 = L1 @ L1.T
    cov2 = L2 @ L2.T
    
    w = 1 / (1 + np.exp(-params[10]))  #* logistic function => weight in (0,1)
    
    return [mean1, mean2], [cov1, cov2], [w, 1 - w]

def objective(params, X, constrain_means=False, constrain_covs=False,
              mean_penalty=1000.0, cov_penalty=100.0):
    """
    Objective function for the optimizer. Minimizes KL divergence and adds additional 
    penalties for violating mean/covariance constraints if requested.
    """
    means, covs, weights = params_to_gmm(params, constrain_means)
    kl_div = estimate_kl_divergence(X, means, covs, weights)
    penalty = 0.0
    
    #* Mean constraint
    if constrain_means:
        y_coord = means[1][1]  #* second component's y-mean
        if y_coord > -10.0 + 0.3 or y_coord < -12.0 - 0.3:
            penalty += mean_penalty * min(abs(y_coord - (-10.0)), abs(y_coord - (-12.0)))
    
    #* Covariance constraint
    if constrain_covs:
        det1 = np.linalg.det(covs[0])
        det2 = np.linalg.det(covs[1])
        if det2 > det1:
            penalty += cov_penalty * (det2 - det1)
    
    return kl_div + penalty

def fit_gmm_to_data(X, constrain_means=False, constrain_covs=False, n_init=10):
    """
    Fit a 2-component GMM to data via KL divergence minimization.
    Multiple random initializations are tried for robustness, and the best solution is chosen.
    fit_results contains:
    'means': list of two 2-element arrays (means for each component)
    'covs': list of two 2x2 arrays (covariance matrices)
    'weights': list of two weights
    'kl_div': KL divergence of the best fit
    ... and additional fields for plotting the GMM density.
    """
    best_result = None
    best_kl = float('inf')
    
    data_cov = np.cov(X.T)
    initializations = []
    
    #* Guided initialization based on median split of y-values
    mid_quantile = np.quantile(X[:, 1], 0.5)
    upper_mask = X[:, 1] > mid_quantile
    lower_mask = ~upper_mask
    
    if np.sum(upper_mask) > 0 and np.sum(lower_mask) > 0:
        mean1 = np.mean(X[upper_mask], axis=0)
        mean2 = np.mean(X[lower_mask], axis=0)
        if constrain_means:
            mean2[1] = np.clip(mean2[1], -12.0, -10.0)
        
        init_params = np.array([
            mean1[0], mean1[1], mean2[0], mean2[1],
            np.log(np.sqrt(data_cov[0, 0])), 0, np.log(np.sqrt(data_cov[1, 1])),
            np.log(np.sqrt(data_cov[0, 0] * 0.7)), 0, np.log(np.sqrt(data_cov[1, 1] * 0.7)),
            0  #* logit(0.5)
        ])
        initializations.append(init_params)
    
    #* Additional random initializations
    for i in range(n_init - len(initializations)):
        idx = np.random.choice(len(X), 2, replace=False)
        mean1 = X[idx[0]]
        mean2 = X[idx[1]]
        if constrain_means:
            mean2[1] = np.random.uniform(-12.0, -10.0)
        
        init_params = np.array([
            mean1[0], mean1[1], mean2[0], mean2[1],
            np.log(np.sqrt(data_cov[0, 0])), 0, np.log(np.sqrt(data_cov[1, 1])),
            np.log(np.sqrt(data_cov[0, 0] * 0.7)), 0, np.log(np.sqrt(data_cov[1, 1] * 0.7)),
            0
        ])
        initializations.append(init_params)
    
    #* Try each initialization
    for i, init_params in enumerate(initializations):
        print(f"Running initialization {i+1}/{len(initializations)}...")
        result = minimize(
            lambda p: objective(p, X, constrain_means, constrain_covs),
            init_params, method='L-BFGS-B', options={'maxiter': 3000, 'disp': False}
        )
        if not result.success:
            print("  Initial optimization failed, trying Nelder-Mead...")
            result = minimize(
                lambda p: objective(p, X, constrain_means, constrain_covs),
                result.x, method='Nelder-Mead', options={'maxiter': 5000, 'disp': False}
            )
        
        means, covs, weights = params_to_gmm(result.x, constrain_means)
        kl_div = estimate_kl_divergence(X, means, covs, weights)
        print(f"  Finished with KL divergence: {kl_div:.4f}")
        
        if kl_div < best_kl:
            best_kl = kl_div
            best_result = (means, covs, weights, kl_div)
            print("  New best result found!")
    
    return best_result


def fit_gmm(original_data, labels=['Star-forming Galaxies', 'Green Valley Galaxies'], Verbose=False):
    """
    Fit a 2-component GMM to 'original_data' using constrained minimization of the KL divergence.
    """
    print("Fitting GMM to original data...")
    means_orig, covs_orig, weights_orig, kl_div_orig = fit_gmm_to_data(
        original_data, constrain_means=True, constrain_covs=True, n_init=5
    )
    print(f"Original data KL divergence: {kl_div_orig:.4f}")
    
    #* Prepare grid for GMM density
    x_min, x_max = original_data[:, 0].min() - 0.5, original_data[:, 0].max() + 0.5
    y_min, y_max = original_data[:, 1].min() - 0.5, original_data[:, 1].max() + 0.5
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    gmm_density_orig = gmm_pdf(grid_points, means_orig, covs_orig, weights_orig).reshape(x_grid.shape)

    if Verbose:
        #* Summary of fitted GMM parameters
        print("\nFitted GMM Parameters for Original Data:")
        for i in range(2):
            print(f"{labels[i]}:")
            print(f"  Weight: {weights_orig[i]:.4f}")
            print(f"  Mean: [{means_orig[i][0]:.4f}, {means_orig[i][1]:.4f}]")
            print(f"  Covariance Matrix:")
            print(f"    [{covs_orig[i][0, 0]:.4f}, {covs_orig[i][0, 1]:.4f}]")
            print(f"    [{covs_orig[i][1, 0]:.4f}, {covs_orig[i][1, 1]:.4f}]")
            print(f"  Determinant: {np.linalg.det(covs_orig[i]):.4f}")
    
    return {
        'means'  : means_orig,
        'covs'   : covs_orig,
        'weights': weights_orig,
        'kl_div' : kl_div_orig,
        'x_grid' : x_grid,
        'y_grid' : y_grid,
        'density': gmm_density_orig,
        'means_orig': means_orig,
        'covs_orig': covs_orig,
        'weights_orig': weights_orig
    }

