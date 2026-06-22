# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS
from dataclasses import dataclass

from scipy.stats import multivariate_normal, fisher_exact, barnard_exact, linregress
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines  # for legend proxies
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


import seaborn as sns

import time 
import re 
from io import StringIO

#* --------------------------------------------------------------------------------
#* Specific plotting librairies imports
#* --------------------------------------------------------------------------------
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt



#* --------------------------------------------------------------------------------
#* Personal librairies imports
#* --------------------------------------------------------------------------------
import sys, os
src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from utils import astro_utils as au
from utils import maths_utils  as mu
from utils import stats_utils  as su
from utils import graphics_utils  as gu
from utils import labels_utils  as lu
from utils import pandas_utils  as pu

#* --------------------------------------------------------------------------------
#* Global variables
#* --------------------------------------------------------------------------------
import config as co

#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
import data_loader as dl
import generate_report as report
import analysis as anl


# endregion

def get_fit(non_quenched, Verbose=True):
    """Fit the GMM used to separate the non-quenched populations."""

    #* --------------------------------------------------------------------------------
    #* Convert data to NumPy arrays (removed PyTorch usage).
    #* Here we keep the raw and optionally normalized versions if needed.
    #* --------------------------------------------------------------------------------
    X = non_quenched[['lgm', 'sSFR']].values

    #* Example optional normalization using NumPy:
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)  #* Add small epsilon to avoid zero div


    #* --------------------------------------------------------------------------------
    #* Set random seed for reproducibility
    #* --------------------------------------------------------------------------------
    np.random.seed(421)



    #* --------------------------------------------------------------------------------
    #* Fit and visualize the GMM on our non-AGN data.
    #* --------------------------------------------------------------------------------
    start_time = time.time()
    fit_results = su.fit_gmm(X)
    if Verbose:
        print(f"Fitting completed in {time.time() - start_time:.2f} seconds")

    return fit_results 

    # NB : Next, can do 
        # su.visualize_gmm(X, fit_results) # Possible file name: 'gmm_fitted_galaxy_population.png'


def get_decision_boundary_interp(non_quenched, fit_results, boundary_margin=0.5, grid_points=200):
    """
    Computes an interpolation function f_interp from the decision boundary
    between star-forming and green valley galaxies.
    
    Parameters
    ----------
    non_quenched : pandas.DataFrame
        DataFrame with columns 'lgm' and 'sSFR'.
    fit_results : dict
        Dictionary with keys 'means', 'covs', and 'weights' from the GMM fit.
    boundary_margin : float, optional
        Extra margin to add/subtract when computing the grid limits.
    grid_points : int, optional
        Number of grid points for both x and y directions.
        
    Returns
    -------
    f_interp : function
        Interpolation function f_interp(mass) that returns the limiting sSFR.
    """
    # Extract GMM parameters.
    means = fit_results['means']
    covs = fit_results['covs']
    weights = fit_results['weights']
    
    
    # Assume the component with the higher sSFR (index 1) is star forming.
    starforming_idx = np.argmax([m[1] for m in means])
    nonstar_idx = 1 - starforming_idx
    
    # Set grid limits.
    x_min = non_quenched['lgm'].min() - boundary_margin
    x_max = non_quenched['lgm'].max() + boundary_margin
    y_min = non_quenched['sSFR'].min() - boundary_margin
    y_max = non_quenched['sSFR'].max() + boundary_margin
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                         np.linspace(y_min, y_max, grid_points))
    grid_points_arr = np.column_stack([xx.ravel(), yy.ravel()])
    
    p_star_grid = weights[starforming_idx] * multivariate_normal.pdf(
        grid_points_arr, mean=means[starforming_idx], cov=covs[starforming_idx], allow_singular=True)
    p_nonstar_grid = weights[nonstar_idx] * multivariate_normal.pdf(
        grid_points_arr, mean=means[nonstar_idx], cov=covs[nonstar_idx], allow_singular=True)
    pdf_diff = p_star_grid - p_nonstar_grid
    pdf_diff = pdf_diff.reshape(xx.shape)
    
    # Draw the zero contour (decision boundary) offscreen.
    fig_temp = plt.figure()
    contour = plt.contour(xx, yy, pdf_diff, levels=[0], colors='black', linestyles='--', linewidths=2)
    plt.close(fig_temp)
    
    # Extract contour coordinates (use the longest segment).
    segments = contour.allsegs[0]
    if len(segments) == 0:
        raise ValueError("No decision boundary contour found.")
    seg = max(segments, key=lambda s: s.shape[0])
    # Sort the segment by mass (first column)
    seg = seg[seg[:, 0].argsort()]
    
    # Create interpolation function: given a mass, return the limiting sSFR.
    f_interp = interp.interp1d(seg[:, 0], seg[:, 1], bounds_error=False, fill_value="extrapolate")
    return f_interp


def flattens_quenched(row):
    """Replace the legacy quenched sentinel by the plotting floor value."""

    # set sSFR to -15 if it is -9999
    if row['sSFR'] == -9999:
        return -15
    else:
        return row['sSFR']


def compute_component_prob(x, comp_idx, fit_results):
    """Evaluate the weighted Gaussian-mixture density of one component."""

    means = fit_results['means']
    covs = fit_results['covs']
    weights = fit_results['weights']
    
    return weights[comp_idx] * multivariate_normal.pdf(x, mean=means[comp_idx], cov=covs[comp_idx], allow_singular=True)


def is_star_forming(cat, fit_results):
    """
    Classify galaxies in a catalogue as star-forming or not based on GMM parameters.
    
    Parameters
    ----------
    cat : pandas.DataFrame
        DataFrame containing the data to classify.
    fit_results : dict
        Dictionary with keys 'means', 'covs', and 'weights' from the GMM fit.
    
    Returns
    -------
    star_forming : list
        List of booleans indicating whether each galaxy is star-forming or not.
    """
    

    star_forming = []

    #* --------------------------------------------------------------------------------
    #* Extract GMM parameters.
    #* --------------------------------------------------------------------------------
    
    means = fit_results['means']
    covs = fit_results['covs']
    weights = fit_results['weights']
    
    #* --------------------------------------------------------------------------------
    #* Assume the component with the higher sSFR (index 1) is star forming.
    #* --------------------------------------------------------------------------------
    
    starforming_idx = np.argmax([m[1] for m in means])
    nonstar_idx = 1 - starforming_idx

    for x in cat[['lgm', 'sSFR']].values:
        p_star = compute_component_prob(x, starforming_idx, fit_results)
        p_nonstar = compute_component_prob(x, nonstar_idx, fit_results)
        star_forming.append(p_star >= p_nonstar)
   
    return star_forming


def sSFR_status(df):
    """
    Classify galaxies based on their sSFR status.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to classify.
    
    Returns
    -------
    None
    """
    
    
    if 'sSFR_status' not in df.columns:
        df['sSFR_status'] = pd.Series(pd.NA, index=df.index, dtype=object)
    current_status = df['sSFR_status']
    blank_mask = current_status.apply(lambda val: isinstance(val, str) and val.strip() == '')
    needs_update = current_status.isna() | blank_mask
    df.loc[needs_update, 'sSFR_status'] = co.sSFR_status[0]  # Default to 'quenched'
    df.loc[needs_update & (df['sSFR'] > -9999), 'sSFR_status'] = co.sSFR_status[1]  # Set to 'passive'
    df.loc[needs_update & (df['is_star_forming']), 'sSFR_status'] = co.sSFR_status[2]  # Set to 'star-forming'

    return df['sSFR_status']

def add_excess(df, f_interp):
    """
    Add the sSFR excess to the dataframes
    """
    df['sSFR_excess'] = df['sSFR'] - f_interp(df['lgm'])
    return df

def add_MS_offset(df, MS_coeffs):
    """
    Add the sSFR main sequence offset to the dataframes
    """
    sSFR_MS = MS_coeffs[0] * df['lgm'] + MS_coeffs[1]
    df['sSFR_MS_offset'] = np.where(
        df['sSFR_status'] == 'Starforming',
        df['sSFR'] - sSFR_MS,
        np.nan
    )
    return df

def add_status(df, fit_results):
    """
    Add the sSFR status to the dataframes
    """
    df['is_star_forming'] = is_star_forming(df, fit_results)
    #    set df['sSFR_status'] as sSFR_status(df) if df['sSFR_status'] is nan else keep df['sSFR_status']
    

    df['sSFR_status'] = sSFR_status(df)
    
    df.drop(columns=['is_star_forming'], inplace=True)
    df['sSFR_raw'] = df['sSFR']
    df['sSFR'] = df.apply(flattens_quenched, axis=1)

    return df
    

def compute_status(sample):
    """Classify all samples in sSFR space and build the decision boundary."""

    non_quenched = sample['SDSS'][sample['SDSS']['sSFR_status'] != co.sSFR_status[0]]
    fit_results = get_fit(non_quenched)
    f_interp = get_decision_boundary_interp(non_quenched, fit_results)
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]+['SDSS']:
        sample[cat] = add_status(sample[cat], fit_results)
    return sample, non_quenched, fit_results, f_interp


def compare(sample, Verbose=True):
    """
    Compare the sSFR of the control samples and the CG sample
    """

    #* --------------------------------------------------------------------------------
    #* Initialising variables
    #* --------------------------------------------------------------------------------
    CG = sample['CG4'+co.GASUFF]
    CG_lgm = CG[CG['lgm'] > 0]
    results = {}
    #* --------------------------------------------------------------------------------
    if Verbose:
            print("CG")
    for status in co.sSFR_status:
        CG_status = CG_lgm[CG_lgm['sSFR_status'] == status]
        frac = len(CG_status) / len(CG_lgm)
        # results = pu.dict_union(results, {df.name : {'sSFR_status': status, 'fraction': f'{100*frac:.1f}\%'}})
        if Verbose:
            print(f"   {status}: {100*frac:.1f} % ")

    for control_name in co.CONTROL:
        if Verbose:
            print(control_name)

        control = sample[control_name+co.GASUFF]
        control_lgm = control[control['lgm'] > 0]
        for status in co.sSFR_status:
            control_status = control_lgm[control_lgm['sSFR_status'] == status]
            frac = len(control_status) / len(control_lgm)
            # results = pu.dict_union(results, {df.name : {'sSFR_status': status, 'fraction': f'{100*frac:.1f}\%'}})
            if Verbose:
                print(f"   {status}: {100*frac:.1f} % ")
        # Calculate the p-value for the proportion of star forming galaxies
        if ((co.sSFR_status[2] in control_lgm['sSFR_status'].unique()) and
            (co.sSFR_status[2] in CG_lgm['sSFR_status'].unique())): # Star forming
            CG_starforming = CG_lgm[CG_lgm['sSFR_status'] == co.sSFR_status[2]]
            control_starforming = control_lgm[control_lgm['sSFR_status'] == co.sSFR_status[2]]
            table = [[len(CG_starforming), len(CG_lgm)],[len(control_starforming), len(control_lgm)]]
            res_fisher = fisher_exact(table, alternative='two-sided')
            # results_old.loc['p_star_forming_diff_Control', df.name] = res_fisher.pvalue
            results = pu.dict_union(results, {control_name+"_"+status+"_vs_CG": res_fisher.pvalue})
            if Verbose:
                print(f"   {status}: {100*len(control_status)/len(control_lgm):.1f} % (among galaxies having a mass)")
                print("Exact test p-values of proportion of star forming being different between CG_4 and the control sample:")
                print(f"   Fisher: {res_fisher.pvalue:.1e}")
                if res_fisher.pvalue < 0.05:
                    print("   Reject null hypothesis: the proportion of star forming galaxies is different between CG_4 and the control sample")
                    print(f"   CG_4 proportion of star forming ({100*len(control_starforming)/len(control_lgm):.1f}%) " + 
                        f"is different from {control_name} proportion of star forming ({100*len(CG_starforming)/len(CG_lgm):.1f}%)")
                else:
                    print("   Fail to reject null hypothesis: the proportion of star forming galaxies is not different between " + 
                        "CG_4 and control sample")

    
    return results
    

def plot_classification(non_quenched, sdss_df, fit_results, f_interp, 
                               fig_size=(12,8), label_fontsize=18, tick_labelsize=16, 
                               legendmarkerscale=5, name=None, quenched_value_set=-15):
    """
    Draws the classification figure:
      - Plots non-quenched galaxies colored by classification (star-forming vs. green valley).
      - Overlays the decision boundary (using f_interp).
      - Plots quenched galaxies (from sdss_df) in red (with their sSFR set to quenched_value_set).
    
    Parameters
    ----------
    non_quenched : pandas.DataFrame
        DataFrame with columns 'lgm' and 'sSFR' for non-quenched galaxies.
    sdss_df : pandas.DataFrame
        DataFrame with SDSS data, must include a column 'sSFR_status'.
    fit_results : dict
        Dictionary with keys 'means', 'covs', 'weights' from the GMM fit.
    f_interp : function
        Interpolation function that returns the limiting sSFR for a given mass.
    fig_size : tuple, optional
        Figure size.
    label_fontsize : int, optional
        Font size for axis labels.
    tick_labelsize : int, optional
        Font size for tick labels.
    pdf_filename : str, optional
        Filename to save the PDF. Use None to skip saving.
    quenched_value_set : float, optional
        Value to assign to sSFR for quenched galaxies.
    """
    # Extract GMM parameters.
    means = fit_results['means']
    covs = fit_results['covs']
    weights = fit_results['weights']
    
    # Decide which component corresponds to star forming.
    starforming_idx = np.argmax([m[1] for m in means])
    nonstar_idx = 1 - starforming_idx
    
    # Compute posterior probabilities for non_quenched galaxies.
    X = non_quenched[['lgm', 'sSFR']].values
    def compute_component_prob(x, comp_idx):
        """Evaluate the local two-component density used for posterior plotting."""

        return weights[comp_idx] * multivariate_normal.pdf(x, mean=means[comp_idx],
                                                             cov=covs[comp_idx],
                                                             allow_singular=True)
    star_forming_list = []
    posterior = []
    for x in X:
        p_star = compute_component_prob(x, starforming_idx)
        p_nonstar = compute_component_prob(x, nonstar_idx)
        if p_star >= p_nonstar:
            star_forming_list.append(True)
            posterior.append(p_star / (p_star + p_nonstar))
        else:
            star_forming_list.append(False)
            posterior.append(p_star / (p_star + p_nonstar))
    
    non_quenched = non_quenched.copy()
    non_quenched['is_star_forming'] = star_forming_list
    non_quenched['posterior_star'] = posterior

    # Create the figure and axes.
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    # Plot non-quenched galaxies: blue for star forming, green for green valley.
    colors = non_quenched['is_star_forming'].map({True: 'blue', False: 'green'})

    for morph in non_quenched['morphology'].unique():
        morph_mask = non_quenched['morphology'] == morph
        ax.scatter(non_quenched[morph_mask]['lgm'], 
                   non_quenched[morph_mask]['sSFR'],
                   s=1, c = lu.morph_color(morph), label=morph)
    # ax.scatter(non_quenched['lgm'], non_quenched['sSFR'],
    #            c=colors, s=1, alpha=0.6) 
    # sns.scatterplot(data=non_quenched, x='lgm', y='sSFR', style='morphology', 
    #                 c=colors, s=1, alpha=0.6, ax=ax)
    

    # Use lu.formatted_label() for axis labels (assumes the module "lu" is imported).
    ax.set_xlabel(lu.formatted_label('lgm'), fontsize=label_fontsize)
    ax.set_ylabel(lu.formatted_label('sSFR'), fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    
    # Plot the decision boundary using f_interp.
    x_vals = np.linspace(non_quenched['lgm'].min()-0.5, 
                         non_quenched['lgm'].max()+0.5, 200)
    y_vals = f_interp(x_vals)
    ax.plot(x_vals, y_vals, 'k--', linewidth=2, label='Star-forming - passive limit')
    
    # Add quenched galaxies as red points.
    # sdss_quenched = sdss_df[sdss_df['sSFR_status'] == 'Q'].copy()
    # sdss_quenched['sSFR'] = quenched_value_set
    # ax.scatter(sdss_quenched['lgm'], sdss_quenched['sSFR'],
    #            c='red', s=5, marker='o', alpha=0.7, label='Quenched (sSFR_status=="Q")')
    
    # Create proxy artists for the legend.
    # boundary_proxy = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2,
    #                                label='Star forming – Passive limit')
    # star_proxy = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=5,
    #                            label='Star Forming')
    # green_proxy = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5,
    #                             label='Passive')
    # red_proxy = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5,
    #                           label='Quenched')
    
    # ax.legend(handles=[boundary_proxy, star_proxy, green_proxy, red_proxy], loc='best')

    ax.legend(markerscale=legendmarkerscale, fontsize=label_fontsize-2, loc='best')
    fig.tight_layout()
    
    if name:
        fig.savefig(co.FIGURES_PATH + name + '.pdf', format='pdf', bbox_inches='tight')
    
    if co.SHOW:
        plt.show()

 
def plot_galaxies(SDSS, CG, markerscale=8, triangle_factor=0.7, name=None, figsize=(10, 8),
                   fontsize_labels=16, fontsize_legend=14, 
                   xmin = 7.5, xmax = 11.8, ymin = co.sSFR_QUENCHED - 0.2, ymax = -8):
    """
    Create a scatter plot of galaxy sSFR vs stellar mass using subplots.
    
    Parameters:
    -----------
    SDSS : pandas DataFrame
        The galaxy SDSS sample containing 'lgm', 'sSFR', and 'sSFR_status' columns
    CG : pandas DataFrame
        The galaxy Compact Groups sample containing 'lgm', 'sSFR', and 'sSFR_status' columns
    markerscale : float, default=8
        Scale factor for the size of markers in the legend for scatter points
    triangle_factor : float, default=0.7
        Factor to make triangles smaller than dots in the legend (relative to markerscale)
    save_path : str, optional
        If provided, save the figure to this path as PDF
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)
    fontsize_labels : int, default=16
        Font size for axis labels
    fontsize_legend : int, default=14
        Font size for legend text
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Filter Control for valid mass values
    # Control = Control.loc[Control['lgm'] > 0]
    
    # Create a copy of the dataframe with capitalized sSFR_status for the legend
    plot_data = SDSS.copy()
    plot_data['sSFR_status'] = plot_data['sSFR_status'].apply(lu.display_label)
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the scatter plot for main Control with capitalized labels
    sns.scatterplot(
        data=plot_data,
        x='lgm',
        y='sSFR',
        hue='sSFR_status',
        palette={
            lu.display_label(co.sSFR_status[0]): 'red',
            lu.display_label(co.sSFR_status[1]): 'green',
            lu.display_label(co.sSFR_status[2]): 'blue',
        },
        alpha=0.5,
        s=1,
        ax=ax,
        legend=False  # Don't create a legend yet
    )
    
    # Create a new legend with proper sizes for dot markers
    dot_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=markerscale, label=lu.display_label(co.sSFR_status[0]), alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=markerscale, label=lu.display_label(co.sSFR_status[1]), alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=markerscale, label=lu.display_label(co.sSFR_status[2]), alpha=0.7)
    ]
    
        
    # Filter CG data for valid mass values
    CG_filtered = CG[CG['lgm'] > 0]
    
    # Plot CG data as empty triangles
    ax.scatter(
        CG_filtered['lgm'],
        CG_filtered['sSFR'],
        edgecolor='black',
        facecolor='none',
        marker='^',
        s=15,  # Size for the actual data points
        alpha=0.7,
        linewidth=1
    )
    
    # Add triangle to legend elements with a smaller size
    triangle_legend_element = plt.Line2D([0], [0], marker='^', color='w', 
                                        markeredgecolor='black', markerfacecolor='none',
                                        markersize=markerscale * triangle_factor, 
                                        label='Compact Groups galaxies', alpha=0.7)
    dot_legend_elements.append(triangle_legend_element)
    
    # Create the legend with our custom elements
    ax.legend(handles=dot_legend_elements, fontsize=fontsize_legend)
    
    # Set axis labels with larger font
    ax.set_xlabel(r'$\log(M_*/M_\odot)$', fontsize=fontsize_labels)
    ax.set_ylabel('sSFR [yr⁻¹]', fontsize=fontsize_labels)

    # Also increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=fontsize_labels-2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if a name is provided
    if name:
        plt.savefig(co.FIGURES_PATH + name + '.pdf', format='pdf', bbox_inches='tight')
    
    return fig, ax


def plot_residual_distribution(non_quenched, f_interp, figsize=(12,8), fontsize=18,
                            name=None):
    """
    Plot the histogram of the vertical residual (galaxy sSFR minus the limiting sSFR).
    
    Parameters
    ----------
    non_quenched : pandas.DataFrame
        DataFrame with columns 'lgm' and 'sSFR'
        for the non-quenched non-AGN galaxies.
    f_interp : function
        Interpolation function that returns the limiting sSFR for a given mass.
    figsize : tuple, optional
        Figure size.
    fontsize : int, optional
        Font size for axis labels.
    pdf_filename : str, optional
        Filename to save the PDF. Set to None to not save.
    """
    # Compute the vertical residual.
    mass_vals = non_quenched['lgm'].values
    limiting_sSFR = f_interp(mass_vals)
    vertical_distance = non_quenched['sSFR'].values - limiting_sSFR
    non_quenched = non_quenched.copy()
    non_quenched['vertical_distance'] = vertical_distance

    # Plot the histogram with Poisson error bars.
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Compute histogram using 50 bins.
    counts, bin_edges = np.histogram(vertical_distance, bins=50)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    error = np.sqrt(counts)  # Poisson error: sqrt(n)

    # Plot histogram as bars.
    bar_width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, counts, width=bar_width, color='tab:blue', alpha=0.7,
           edgecolor='black', align='center')
    # Plot error bars.
    ax.errorbar(bin_centers, counts, yerr=error, fmt='none', ecolor='black', capsize=2)

    # Set labels.
    ax.set_xlabel("Residual sSFR", fontsize=fontsize)
    ax.set_ylabel("Number of galaxies", fontsize=fontsize)
    # Set y-axis to log scale.
    ax.set_yscale("log")

    # Format y-axis tick labels in scientific notation (LaTeX style).
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: r'$%s$' % format(x, '.0e')))
    ax.tick_params(axis='both', labelsize=12)

    # enlarge axis labels
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    # switch y labels to 1, 10, 100, 1000
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # ax.yaxis.get_major_formatter().set_scientific(True)
    # ax.yaxis.get_major_formatter().set_powerlimits((-1,1))

    fig.tight_layout()

    # Save figure if a name is provided
    if name:
        plt.savefig(co.FIGURES_PATH + name + '.pdf', format='pdf', bbox_inches='tight')
    
    if co.SHOW:
        plt.show()


def plot_density_original_vs_GMMfit(X, fit_results, figsize=(16, 8), dpi=150, name=None):
    """
    Create a side-by-side density plot of the original data and the fitted GMM.
    """

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


    def plot_gaussian_contours(ax, mean, cov, color, alpha=0.3):
        """
        Draw an ellipse representing a specified covariance contour (3 sigma) 
        for clarity in GMM component distribution plots.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals) * 3
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                        facecolor=color, alpha=alpha, edgecolor='black')
        ax.add_patch(ellipse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    counts, x_edges, y_edges = np.histogram2d(
        X[:, 0], X[:, 1],
        bins=100, density=True
    )

    #* Plot 1: Density of the original data
    im1 = ax1.pcolormesh(
        x_edges, y_edges, counts.T,
        cmap='viridis',
        norm=LogNorm(vmin=max(0.01, counts.min()), vmax=counts.max())
    )
    ax1.set_title('Original Non-AGN Galaxy Data Density', fontsize=16, fontweight='bold')
    ax1.set_xlabel(r'$\log_{10}(M_*)$ [Solar masses]', fontsize=14)
    ax1.set_ylabel(r'$\log_{10}(\mathrm{sSFR})$ [yr$^{-1}$]', fontsize=14)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Probability Density', fontsize=12)

    #* Plot 2: GMM density
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    gmm_density = gmm_pdf(grid_points, fit_results['means'], fit_results['covs'], fit_results['weights']).reshape(x_grid.shape)

    im2 = ax2.pcolormesh(
        x_grid, y_grid, gmm_density,
        cmap='viridis',
        norm=LogNorm(vmin=max(0.01, gmm_density.min()), vmax=gmm_density.max())
    )
    contour_levels = np.logspace(np.log10(gmm_density.max()/100), np.log10(gmm_density.max()/1.5), 5)
    ax2.contour(
        x_grid, y_grid, gmm_density,
        levels=contour_levels, colors='white', alpha=0.5, linewidths=1.0
    )

    #* Plot GMM component ellipses
    for i in range(2):
        plot_gaussian_contours(
            ax2, fit_results['means'][i], fit_results['covs'][i], 
            color='red' if i == 0 else 'blue', alpha=0.2
        )

    ax2.set_title('Fitted Two-Component GMM Density', fontsize=16, fontweight='bold')
    ax2.set_xlabel(r'$\log_{10}(M_*)$ [Solar masses]', fontsize=14)
    ax2.set_ylabel(r'$\log_{10}(\mathrm{sSFR})$ [yr$^{-1}$]', fontsize=14)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('GMM Probability Density', fontsize=12)

    #* Add component annotations
    from matplotlib.patches import Patch
    component_labels = []
    for i in range(2):
        weight = fit_results['weights'][i]
        component_labels.append(f"Component {i+1} (w={weight:.2f})")

    legend_elements = [
        Patch(facecolor='red', alpha=0.2, edgecolor='black', label=component_labels[0]),
        Patch(facecolor='blue', alpha=0.2, edgecolor='black', label=component_labels[1]),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', framealpha=1.0)

    #* Display KL divergence
    ax2.text(
        0.05, 0.95, f"KL Divergence: {fit_results['kl_div']:.4f}",
        transform=ax2.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )

    plt.tight_layout()
    # Save figure if a name is provided
    if name:
        plt.savefig(co.FIGURES_PATH + name + '.pdf', format='pdf', bbox_inches='tight')
    
    if co.SHOW:
        plt.show()
 
    plt.close()


def restrict_analysis(df, df_name, restric_name):
    """Count galaxies by sSFR class inside a restricted subsample and report them."""

    if co.VERBOSE:
        print(df_name)
    total = len(df)
    results = {}
    for status in co.sSFR_status:
        n_df = len(df[df['sSFR_status'] == status])
        results[status] = n_df
        report.append_json(f'{df_name}_{restric_name}_N{status}', n_df)
        report.append_json(f'{df_name}_{restric_name}_N{status}_pc', f"{100*n_df/total:.1f}")
        if co.VERBOSE:
            print(f".  {status}: {n_df} / {total} = {n_df/total:.1f}")

    return results


def pval_restrict_analysis(res1, res2, df1_name, df2_name, restric_name):
    """Compare two restricted samples with a Fisher exact test on star-forming counts."""

    matrix = [[res1[co.sSFR_status[1]], res1[co.sSFR_status[2]]],
                  [res2[co.sSFR_status[1]], res2[co.sSFR_status[2]]]]
            
    res_fisher = fisher_exact(matrix, alternative='two-sided')
    report.append_json(
        f'{restric_name}_star_forming_pvalue_{df2_name}_vs_{df1_name}',
        gu.pvalue_latex(res_fisher.pvalue),
    )
    if co.VERBOSE:
        print(f"Exact test p-values of proportion of star forming {restric_name} being different between {df1_name} and {df2_name}:")
        print(f"   Fisher: {res_fisher.pvalue:.1e}")
        if res_fisher.pvalue < 0.05:
            print("   Reject null hypothesis: the proportion is different")
        else:
            print("   Fail to reject null hypothesis: the proportion is not different")


def BGGs_analysis(sample):
    """
    Analyze the sSFR status of Brightest Group Galaxies (BGGs) in compact groups.
    
    Parameters
    ----------
    sample : dict
        Dictionary containing dataframes for different samples, including 'CG4' compact groups.
    """

    report.append_json('BGG_sSFR_tests', 'two-sided Fisher exact test')

    CG4 = sample['CG4'+co.GASUFF]
    restrict_CG4 = {}
    restrict_CG4['BGG'] = CG4[CG4['rank_M'] == 1]
    restrict_CG4['Sat'] = CG4[CG4['rank_M'] > 1] 

    results_CG4 = {}
    for rest_type in restrict_CG4.keys():
        results_CG4[rest_type] = restrict_analysis(restrict_CG4[rest_type], 'CG4', rest_type)

    
    for cat in co.CONTROL.keys():
        df = sample[cat+co.GASUFF] 
        BGG = df[df['rank_M'] == 1]
        Sat = df[df['rank_M'] > 1]
        restrict_df = {'BGG': BGG, 'Sat': Sat}
        for rest_type in restrict_df.keys():
            results_df = restrict_analysis(restrict_df[rest_type], cat, rest_type)
            pval_restrict_analysis(results_CG4[rest_type], results_df, 'CG4', cat, rest_type)


def correlations_by_fertility(sample):
    """
    Analyze correlations between sSFR status and fertility in galaxies.
    
    Parameters
    ----------
    sample : dict
        Dictionary containing dataframes for different samples, including 'CG4' compact groups.
    """

    report.append_json('Fertility_sSFR_tests', 'two-sided Fisher exact test')

    CG4 = sample['CG4'+co.GASUFF]
    restrict_CG4 = {}
    restrict_CG4['Fertile'] = CG4[CG4['fertility'] == 'Fertile']
    restrict_CG4['Sterile'] = CG4[CG4['fertility'] == 'Sterile'] 

    results_CG4 = {}
    for rest_type in restrict_CG4.keys():
        results_CG4[rest_type] = restrict_analysis(restrict_CG4[rest_type], 'CG4', rest_type)

    
    for cat in co.CONTROL.keys():
        df = sample[cat+co.GASUFF] 
        Fertile = df[df['fertility'] == 'Fertile']
        Sterile = df[df['fertility'] == 'Sterile']
        restrict_df = {'Fertile': Fertile, 'Sterile': Sterile}
        for rest_type in restrict_df.keys():
            results_df = restrict_analysis(restrict_df[rest_type], cat, rest_type)
            pval_restrict_analysis(results_CG4[rest_type], results_df, 'CG4', cat, rest_type)

 

def split_by_fertility(sample, make_plots=True,
                       plot_name="ssfr_class",
                       figsize=(10, 12),
                       label_fontsize=18,
                       tick_labelsize=16):
    """Compare galaxy properties after splitting each sample by sSFR class."""

    def local_prec(val, base_prec=4):
        """Use slightly lower precision for order-unity values in the JSON output."""

        if np.abs(val) < 10:
            return base_prec - 1
        else:
            return base_prec

    QUANTITIES_FERTILITY = ['sSFR', 'M_r', 'lgm']
    QUANTITIES_PLOT = ['M_r', 'lgm']

    if co.VERBOSE:
        print("Analyzing correlations by fertility...")

    plot_rows = []

    for name in co.SAMPLE.keys():
        if co.VERBOSE:
            print(f"   Analyzing {name}...")

        cat = name + co.GASUFF

        # Split by galaxy's own fertility
        split = {
            co.sSFR_status[index]: sample[cat][
                sample[cat]['sSFR_status'] == co.sSFR_status[index]
            ]
            for index in [1, 2]
        }

        stats = anl.stats_comp_split(split)

        # ================================================================
        # A) Store per-population medians + errors (already working)
        # ================================================================
        for part in split.keys():  # Passive / Starforming
            for quantity in QUANTITIES_FERTILITY:
                median_val = stats[part][quantity]

                report.append_json(
                    f'{name}_{part}_{quantity}_median',
                    gu.numformat(median_val, prec=local_prec(median_val))
                )

                median_hat, sigma_median, ci_low, ci_high = su.bootstrap_median_error(
                    split[part][quantity]
                )

                report.append_json(
                    f'{name}_{part}_{quantity}_median_err',
                    f'{sigma_median:.2f}'
                )

                report.append_json(
                    f'{name}_{part}_{quantity}_median_ci_low',
                    gu.numformat(ci_low, prec=local_prec(ci_low))
                )
                report.append_json(
                    f'{name}_{part}_{quantity}_median_ci_high',
                    gu.numformat(ci_high, prec=local_prec(ci_high))
                )

            # Store counts (unchanged)
            for status, count in stats[part]['sSFR_status_counts'].items():
                report.append_json(f'{name}_{part}_sSFR_status_{status}_N', count)
            for morpho, count in stats[part]['morphology_counts'].items():
                report.append_json(f'{name}_{part}_morphology_{morpho}_N', count)

        # ================================================================
        # B) NEW : Difference Passive – Starforming (per sample)
        # ================================================================
        if co.sSFR_status[1] in split and co.sSFR_status[-1] in split:
            A = split[co.sSFR_status[1]] # Passive
            B = split[co.sSFR_status[-1]] # Starforming

            for quantity in QUANTITIES_FERTILITY:
                diff_hat, sigma_diff, prob = su.bootstrap_median_diff_probability(
                    A[quantity],
                    B[quantity]
                )

                report.append_json(
                    f'{name}_Diff_{quantity}_median',
                    gu.numformat(diff_hat, prec=local_prec(diff_hat))
                )

                report.append_json(
                    f'{name}_Diff_{quantity}_median_err',
                    f'{sigma_diff:.2f}'
                )

                report.append_json(
                    f'{name}_Diff_{quantity}_prob_diff',
                    f'{prob:.3f}'
                )

        # ================================================================
        # C) Add data for violin plots (unchanged)
        # ================================================================
        if make_plots:
            for part in split.keys():
                df_part = split[part][['M_r', 'lgm']].copy()
                df_part['Sample'] = name
                df_part['fertility'] = part
                plot_rows.append(df_part)

    # ------------------------------------------------------------
    # D) Produce violin plot (same as before)
    # ------------------------------------------------------------
    # <--- keep your current plotting code exactly as in BGG version --->
    # ------------------------------------------------------------
    #  VIOLIN PLOTS — ONE COLUMN (N rows × 1 column)
    # ------------------------------------------------------------
    if make_plots and len(plot_rows) > 0:
        df_plot = pd.concat(plot_rows, ignore_index=True)

        sample_order = list(co.SAMPLE.keys())
        fertility_order = [co.sSFR_status[1], co.sSFR_status[2]]  # Passive, Starforming

        quantities = QUANTITIES_PLOT

        sns.set_style("whitegrid")

        n_quant = len(quantities)
        nrows = n_quant
        ncols = 1  # one column

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True
        )

        # axes is a single Axes object if nrows == 1
        if nrows == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel()

        for i, quantity in enumerate(quantities):
            ax = axes[i]

            sns.violinplot(
                data=df_plot,
                x="Sample",
                y=quantity,
                hue="fertility",
                hue_order=fertility_order,
                order=sample_order,
                split=True,
                cut=0,
                density_norm='width',
                inner="quart",
                ax=ax,
            )

            if quantity == "M_r":
                ax.invert_yaxis()

            ax.set_xlabel("")
            ax.set_ylabel(lu.formatted_label(quantity), fontsize=label_fontsize)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=tick_labelsize)

            # Remove local legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Global legend in the top-right corner
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [lu.display_label(label) for label in labels]
        fig.legend(
            handles,
            labels,
            title="sSFR class",
            title_fontsize=label_fontsize,
            loc="upper right",
            bbox_to_anchor=(0.8, 0.99),
            fontsize=label_fontsize - 2
        )

        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

        # ------------------------------------------------------------
        # Saving using your exact pattern
        # ------------------------------------------------------------
        if plot_name:
            filepath = co.FIGURES_PATH + plot_name + '.pdf'
            fig.savefig(filepath, format='pdf', bbox_inches='tight')

        if co.SHOW:
            plt.show()

        plt.close(fig)



def split_by_BGG_fertility(sample, make_plots=True,
                           plot_name="BGG_ssfr_class", figsize=(10, 12),
                           label_fontsize=18, tick_labelsize=16):
    """Compare group members by the sSFR class of their BGG."""

    def local_prec(val, base_prec=4):
        """Use slightly lower precision for order-unity values in the JSON output."""

        if np.abs(val) < 10:
            return base_prec - 1
        else:
            return base_prec

    if co.VERBOSE:
        print("Analyzing correlations by BGG fertility...")

    QUANTITIES_BGG = [
        'sSFR', 'M_r', 'lgm'
    ]
    # --- Collect data for plotting across all samples ---
    plot_rows = []

    for name in co.SAMPLE.keys():
        if co.VERBOSE:
            print(f"   Analyzing {name}...") 

        cat = name + co.GASUFF

        # BGG_split: only BGGs (rank_M == 1), split by fertility
        BGG_split = {
            co.sSFR_status[index]: sample[cat][
                (sample[cat]['sSFR_status'] == co.sSFR_status[index]) &
                (sample[cat]['rank_M'] == 1)
            ]
            for index in [1, 2]
        }

        # split: all group members whose BGG is of given fertility
        split = {
            co.sSFR_status[index]: sample[cat][
                sample[cat]['Group'].isin(
                    BGG_split[co.sSFR_status[index]]['Group']
                )
            ]
            for index in [1, 2]
        }

        # Stats for all desired quantities
        stats = anl.stats_comp_split(split)
        for part in split.keys():  # e.g. 'Passive', 'Starforming'
            for quantity in QUANTITIES_BGG:
                report.append_json(
                    f'{name}_BGG_{part}_{quantity}_median',
                    gu.numformat(
                        stats[part][quantity],
                        prec=local_prec(stats[part][quantity])
                    )
                )
                # bootstrap error bar on median
                median_hat, sigma_median, ci_low, ci_high = su.bootstrap_median_error(split[part][quantity])
                report.append_json(
                    f'{name}_BGG_{part}_{quantity}_median_err',
                    # gu.numformat(sigma_median, prec=local_prec(sigma_median))
                    f'{sigma_median:.2f}'
                )
                # optional: CI bounds if you want to store them
                report.append_json(
                    f'{name}_BGG_{part}_{quantity}_median_ci_low',
                    gu.numformat(ci_low, prec=local_prec(ci_low))
                )
                report.append_json(
                    f'{name}_BGG_{part}_{quantity}_median_ci_high',
                    gu.numformat(ci_high, prec=local_prec(ci_high))
                )

            # --- Collect raw values for plotting ---
            if make_plots:
                df_part = split[part][QUANTITIES_BGG].copy()
                df_part['Sample'] = name
                df_part['BGG_fertility'] = part
                plot_rows.append(df_part)

    # ------------------------------------------------------------
    #  VIOLIN PLOTS — ONE COLUMN (N rows × 1 column)
    # ------------------------------------------------------------
    if make_plots and len(plot_rows) > 0:
        df_plot = pd.concat(plot_rows, ignore_index=True)

        sample_order = list(co.SAMPLE.keys())
        fertility_order = [co.sSFR_status[1], co.sSFR_status[2]]  # Passive, Starforming

        quantities = QUANTITIES_BGG

        sns.set_style("whitegrid")

        n_quant = len(quantities)
        nrows = n_quant
        ncols = 1  # <<< ONE COLUMN

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True
        )

        # axes is a single Axes object if nrows == 1
        if nrows == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel()

        for i, quantity in enumerate(quantities):
            ax = axes[i]

            sns.violinplot(
                data=df_plot,
                x="Sample",
                y=quantity,
                hue="BGG_fertility",
                hue_order=fertility_order,
                order=sample_order,
                split=True,
                cut=0,
                density_norm='width',
                inner="quart",
                ax=ax,
            )

            if quantity == "M_r":
                ax.invert_yaxis()

            ax.set_xlabel("")
            ax.set_ylabel(lu.formatted_label(quantity), fontsize=label_fontsize)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=tick_labelsize)

            # Remove local legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Global legend in the top-right corner
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [lu.display_label(label) for label in labels]
        fig.legend(
            handles,
            labels,
            title="BGG sSFR class",
            title_fontsize=label_fontsize,
            loc="upper right",
            bbox_to_anchor=(0.8, 0.99),
            fontsize=label_fontsize-2
        )

        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

        # ------------------------------------------------------------
        # Saving using your exact pattern
        # ------------------------------------------------------------
        if plot_name:
            filepath = co.FIGURES_PATH + plot_name + '.pdf'
            fig.savefig(filepath, format='pdf', bbox_inches='tight')

        if co.SHOW:
            plt.show()

        plt.close(fig)



def satellites_split_by_BGG_fertility(sample, make_plots=True,
                                      plot_name="Satellites_by_BGG_ssfr_class", figsize=(10, 12),
                                      label_fontsize=18, tick_labelsize=16):
    """Compare only satellites after splitting groups by their BGG sSFR class."""

    def local_prec(val, base_prec=4):
        """Use slightly lower precision for order-unity values in the JSON output."""

        if np.abs(val) < 10:
            return base_prec - 1
        else:
            return base_prec

    if co.VERBOSE:
        print("Analyzing correlations by BGG fertility...")

    QUANTITIES_BGG = [
        'sSFR', 'M_r', 'lgm'
    ]
    # --- Collect data for plotting across all samples ---
    plot_rows = []

    for name in co.SAMPLE.keys():
        if co.VERBOSE:
            print(f"   Analyzing {name}...") 

        cat = name + co.GASUFF

        # BGG_split: only BGGs (rank_M == 1), split by fertility
        BGG_split = {
            co.sSFR_status[index]: sample[cat][
                (sample[cat]['sSFR_status'] == co.sSFR_status[index]) &
                (sample[cat]['rank_M'] == 1)
            ]
            for index in [1, 2]
        }

        # split: all group members whose BGG is of given fertility
        split = {
            co.sSFR_status[index]: sample[cat][
                sample[cat]['Group'].isin(BGG_split[co.sSFR_status[index]]['Group']) &
                (sample[cat]['rank_M'] > 1)  # Only satellites)
            ]
            for index in [1, 2]
        }

        # Stats for all desired quantities
        stats = anl.stats_comp_split(split)
        for part in split.keys():  # e.g. 'Passive', 'Starforming'
            for quantity in QUANTITIES_BGG:
                report.append_json(
                    f'{name}_Sat_BGG_{part}_{quantity}_median',
                    gu.numformat(
                        stats[part][quantity],
                        prec=local_prec(stats[part][quantity])
                    )
                )
                # bootstrap error bar on median
                median_hat, sigma_median, ci_low, ci_high = su.bootstrap_median_error(split[part][quantity])
                report.append_json(
                    f'{name}_Sat_BGG_{part}_{quantity}_median_err',
                    # gu.numformat(sigma_median, prec=local_prec(sigma_median))
                    f'{sigma_median:.2f}'
                )
                # optional: CI bounds if you want to store them
                report.append_json(
                    f'{name}_Sat_BGG_{part}_{quantity}_median_ci_low',
                    gu.numformat(ci_low, prec=local_prec(ci_low))
                )
                report.append_json(
                    f'{name}_Sat_BGG_{part}_{quantity}_median_ci_high',
                    gu.numformat(ci_high, prec=local_prec(ci_high))
                )

            # --- Collect raw values for plotting ---
            if make_plots:
                df_part = split[part][QUANTITIES_BGG].copy()
                df_part['Sample'] = name
                df_part['BGG_fertility'] = part
                plot_rows.append(df_part)

    # ------------------------------------------------------------
    #  VIOLIN PLOTS — ONE COLUMN (N rows × 1 column)
    # ------------------------------------------------------------
    if make_plots and len(plot_rows) > 0:
        df_plot = pd.concat(plot_rows, ignore_index=True)

        sample_order = list(co.SAMPLE.keys())
        fertility_order = [co.sSFR_status[1], co.sSFR_status[2]]  # Passive, Starforming

        quantities = QUANTITIES_BGG

        sns.set_style("whitegrid")

        n_quant = len(quantities)
        nrows = n_quant
        ncols = 1  # <<< ONE COLUMN

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True
        )

        # axes is a single Axes object if nrows == 1
        if nrows == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel()

        for i, quantity in enumerate(quantities):
            ax = axes[i]

            sns.violinplot(
                data=df_plot,
                x="Sample",
                y=quantity,
                hue="BGG_fertility",
                hue_order=fertility_order,
                order=sample_order,
                split=True,
                cut=0,
                density_norm='width',
                inner="quart",
                ax=ax,
            )

            if quantity == "M_r":
                ax.invert_yaxis()

            ax.set_xlabel("")
            ax.set_ylabel(lu.formatted_label(quantity), fontsize=label_fontsize)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=tick_labelsize)

            # Remove local legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Global legend in the top-right corner
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [lu.display_label(label) for label in labels]
        fig.legend(
            handles,
            labels,
            title="Satellites by BGG sSFR class",
            title_fontsize=label_fontsize,
            loc="upper right",
            bbox_to_anchor=(0.8, 0.99),
            fontsize=label_fontsize-2
        )

        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

        # ------------------------------------------------------------
        # Saving using your exact pattern
        # ------------------------------------------------------------
        if plot_name:
            filepath = co.FIGURES_PATH + plot_name + '.pdf'
            fig.savefig(filepath, format='pdf', bbox_inches='tight')

        if co.SHOW:
            plt.show()

        plt.close(fig)





from sklearn.linear_model import LinearRegression

def compute_main_sequence(
        df_sample,
        fert_col="sSFR_status",
        sSFR_col="sSFR",
        mass_col="lgm"):
    """
    Computes Δlog(sSFR) residuals for *one* sample.

    Steps:
      - Select star-forming galaxies
      - Fit log_sSFR = a * lgm + b
      - Compute residuals for ALL galaxies in this df
    Returns:
      df_out : dataframe with new column 'sSFR_residual'
      (a, b) : slope and intercept of SF fit
    """

    # -----------------------------------------
    # Select the star-forming (SF) subsample
    # -----------------------------------------
    sf = df_sample[df_sample[fert_col] == co.sSFR_status[-1]] # Starforming

    if len(sf) < 3:
        return None

    # -----------------------------------------
    # Linear regression for SF galaxies
    # -----------------------------------------
    X = sf[[mass_col]].values.reshape(-1, 1)
    y = sf[sSFR_col].values

    reg = LinearRegression()
    reg.fit(X, y)

    return reg



@dataclass(frozen=True)
class PolynomialModel1D:
    coeffs: np.ndarray   # highest degree first

    def predict(self, x: np.ndarray | pd.Series) -> np.ndarray:
        """Evaluate the polynomial model on one or many x values."""

        x = np.asarray(x, dtype=float)
        return np.polyval(self.coeffs, x)

    def residuals(
        self,
        df: pd.DataFrame,
        x_col="lgm",
        y_col="sSFR",
    ) -> np.ndarray:
        """Return observed minus model-predicted values for a dataframe."""

        return df[y_col].to_numpy(dtype=float) - self.predict(df[x_col])

def fit_ssfr_vs_lgm_poly(
    df: pd.DataFrame,
    order: int = 2,
    x_col="lgm",
    y_col="sSFR",
) -> PolynomialModel1D:
    """Fit a polynomial main-sequence model in the sSFR-mass plane."""

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    coeffs = np.polyfit(x[m], y[m], deg=order)

    return PolynomialModel1D(coeffs=coeffs)



def plot_main_sequence_models(_df, labelsize = 14, ticksize = 12, figname = "Main_Sequence_polyfits"):
    """Plot polynomial main-sequence fits of increasing order on the SDSS sample."""
    
    
    # --- data ---
    x = _df["lgm"].to_numpy(dtype=float)
    y = _df["sSFR"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    xg = np.linspace(x.min(), x.max(), 400)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --------------------------------------------------
    # KDE contours (log scale)
    # --------------------------------------------------
    sns.kdeplot(
        x=x,
        y=y,
        levels=8,
        fill=False,
        linewidths=1.2,
        # cmap="Greys",
        cmap="Blues",
        log_scale=(False, False),   # keep axes linear
        ax=ax,
    )

    # --------------------------------------------------
    # Scatter (light)
    # --------------------------------------------------
    # ax.scatter(x, y, s=0.2, alpha=0.6, color="gray", zorder=1)
    ax.scatter(
        x, y,
        s=4,                 # increase slightly so dots are visible
        marker=".",          # explicit point marker
        alpha=0.6,
        color="gray",
        linewidths=0,        # no edges
        zorder=1,
    )
    # --------------------------------------------------
    # Polynomial fits + χ²
    # --------------------------------------------------
    N = len(x)

    for order in range(1, 5):
        coeffs = np.polyfit(x, y, deg=order)
        y_hat = np.polyval(coeffs, x)
        resid = y - y_hat

        chi2 = np.sum(resid**2)      # σ = 1 assumption
        dof = N - (order + 1)
        red_chi2 = chi2 / dof if dof > 0 else np.nan

        yg = np.polyval(coeffs, xg)
        ax.plot(
            xg,
            yg,
            linestyle="--",
            linewidth=1.5,
            label = fr"Fit to order {order}: $\chi^2_r = {red_chi2:.2f}$",
            zorder=5,
        )
        if order == 1:
            coeffs_1 = coeffs
            report.append_json('Main_Sequence_polyfit_order1_coeffs', [f'{c:.4f}' for c in coeffs])
            report.append_json('Main_Sequence_polyfit_order1_reduced_chi2', f'{red_chi2:.4f}')

    ax.tick_params(
        axis="both",   
        which="major",
        labelsize=ticksize,
    )
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$",fontsize=labelsize)
    ax.set_ylabel(r"$\log(\mathrm{sSFR})$",fontsize=labelsize)
    ax.legend()
    plt.tight_layout()

    if figname:
        plt.savefig(co.FIGURES_PATH + figname + '.pdf', format='pdf', bbox_inches='tight')

    if co.SHOW:
        plt.show()
    
    return coeffs_1

def add_MS_residuals(
    sample: dict,
    model,
    suffix: str,
    x_col="lgm",
    y_col="sSFR",
    status_col="sSFR_status",
    sf_value=co.sSFR_status[-1],
    out_col="MS_res",
    non_sf_value=np.nan,   # could be None, np.nan, -99, etc.
):
    """
    Add MS residuals to all dataframes whose key ends with `suffix`.
    """

    for key, df in sample.items():
        if not key.endswith(suffix):
            continue

        # Work on a copy only if needed
        # df = df.copy()

        is_sf = df[status_col] == sf_value

        # initialise column
        df[out_col] = non_sf_value

        # compute residuals only for star-forming galaxies
        y_hat = model.predict(df.loc[is_sf, x_col])
        df.loc[is_sf, out_col] = (
            df.loc[is_sf, y_col].to_numpy(dtype=float) - y_hat
        )

def plot_main_sequence_residuals(
    sample: dict,
    figname: str | None = None,
    suffix: str | None = None,
    res_col: str = "MS_res",
    bins: int | np.ndarray = 40,
    range: tuple[float, float] | None = None,
    density: bool = True,
    figsize: tuple[float, float] = (8, 5),
    xlabel: str = r"$\Delta \log(\mathrm{sSFR})$",
    ylabel: str | None = None,
    labelsize: int = 14,
    ticksize: int = 12,
    legendsize: int = 10,
    linewidth: float = 1.6,
    alpha: float = 0.65,
    title: str | None = None,
    show: bool | None = None,
):
    """Plot the main-sequence residual distributions for each galaxy sample."""

    if suffix is None:
        suffix = co.GASUFF
    if show is None:
        show = co.SHOW

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (key, df) in enumerate(sample.items()):
        if not str(key).endswith(suffix):
            continue
        if res_col not in df.columns:
            continue

        res = df[res_col].to_numpy()
        res = res[np.isfinite(res)]
        if res.size == 0:
            continue

        bins = np.linspace(-1.1, 1.1, 10)

        ax.hist(
            res,
            bins=bins,
            range=range,
            density=density,
            histtype="step",          # ← journal standard
            linewidth=linewidth,
            color=colors[i % len(colors)],
            alpha=alpha,
            label=lu.display_label(str(key)),
        )

        ax.axvline(
            np.nanmedian(res),
            color=colors[i % len(colors)],
            linestyle="--",
            linewidth=1.2,
        )

    ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is None:
        ylabel = "Probability density" if density else "Number"
    ax.set_ylabel(ylabel, fontsize=labelsize)

    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=labelsize)

    ax.legend(
        fontsize=legendsize,
        frameon=False,     # ← very important
        loc="best",
    )

    if figname is not None:
        plt.savefig(
            co.FIGURES_PATH + figname + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def compare_main_sequence_residuals_bootstrap(sample):
    """ 
    Compare main sequence residuals between CG4 and other samples using bootstrap.
    Parameters
    ----------
    sample : dict
        Dictionary containing dataframes for different samples, including 'CG4' compact groups.

    Returns
    -------
    results : dict
        Dictionary with keys as sample names and values as dictionaries containing:
            - 'Δmedian': Difference in medians (CG4 - other)
            - 'CI_16': 16th percentile of the bootstrap distribution
            - 'CI_84': 84th percentile of the bootstrap distribution
            - 'CI_95_low': 2.5th percentile of the bootstrap distribution
            - 'CI_95_high': 97.5th percentile of the bootstrap distribution
            - 'p_value': p-value for the hypothesis that CG4 median is greater than other sample median
    """

    cg4_key = "CG4" + co.GASUFF
    cg4 = sample[cg4_key]["MS_res"].to_numpy()
    cg4 = cg4[np.isfinite(cg4)]

    results = {}

    for key, df in sample.items():
        if not key.endswith(co.GASUFF):
            continue
        if key == cg4_key:
            continue

        other = df["MS_res"].to_numpy()
        other = other[np.isfinite(other)]

        delta, lo, hi, p, lo95, hi95 = su.bootstrap_median_difference(
            cg4, other, random_state=20260612, return_ci95=True
        )

        results[key] = {
            "Δmedian": delta,
            "CI_16": lo,
            "CI_84": hi,
            "CI_95_low": lo95,
            "CI_95_high": hi95,
            "interval_16_84_level": 0.68,
            "delta_sign_convention": "median(CG4) - median(control)",
            "p_value_method": "two-sided bootstrap sign probability for the median difference",
            "p_value": p,
        }

    return results
