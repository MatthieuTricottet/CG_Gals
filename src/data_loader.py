# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS

from scipy.stats import multivariate_normal
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines  # for legend proxies
import matplotlib.ticker as ticker

import seaborn as sns

import time as time
import re as re
from io import StringIO

import config as co

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

#* --------------------------------------------------------------------------------
#* Global variables
#* --------------------------------------------------------------------------------
import config as co

#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
import generate_report as report

#* --------------------------------------------------------------------------------
#* Module specific imports
#* --------------------------------------------------------------------------------
from astroquery.sdss import SDSS


# endregion

# region variables
gal_suff = "_Gals"
group_suff = "_Groups"
sample_list = ["CG4","Control4B","Control4C","RG4"]
sample_list_name = {"CG4" : r"\CG","Control4B":r"\CB","Control4C":r"\CC","RG4":r"\RG"}
control_list = [samp for samp in sample_list if samp not in ['CG4']]
control_list_name = {samp : sample_list_name[samp] for samp in control_list}
Gals_list = ["{}{}".format(i,gal_suff) for i in sample_list]
Groups_list = ["{}{}".format(i,group_suff) for i in sample_list]
# endregion


def load_previous_samples():

    my_sample={}

    for file in Gals_list+Groups_list:
        my_sample[file] = pd.read_csv(co.DATA_PATH+file+".csv")
        if co.VERBOSE:
            print(f"{len(my_sample[file])} objects loaded from {file}")

    return my_sample

    
def remove_split_CG(Gals,Groups):
    """ Remove the CGs that have been split in the parent catalogue
        cf. Zheng & Shen

        Parameters
        ----------
        Gals : pd.DataFrame
            DataFrame containing the galaxies of the CG sample
        Groups : pd.DataFrame
            DataFrame containing the groups of the CG sample

        Returns
        -------
        Gals_clean : pd.DataFrame
            Cleaned DataFrame containing the galaxies of the CG sample
        Groups_clean : pd.DataFrame
            Cleaned DataFrame containing the groups of the CG sample
    """


    Groups_clean = Groups.loc[~(Groups['Class']=='Split')].reset_index(drop=True)
    Gals_clean = Gals.loc[Gals['Group'].isin(Groups_clean['Group'])].reset_index(drop=True)

    return Gals_clean, Groups_clean


def remove_AGN(df):
#* --------------------------------------------------------------------------------
#* Remove AGN from df
#* --------------------------------------------------------------------------------
    """
    Remove AGN from df.
    """
    df = df[~df['is_AGN']].copy()
    return df

def sSFR_floor(cat, sSFR="sSFR"):
    # if cat[sSFR] < co.sSFR_THRESHOLD, replace by co.sSFR_QUENCHED
    cat.loc[cat[sSFR]<co.sSFR_THRESHOLD, 'sSFR_status'] = co.sSFR_status[0]
    cat.loc[cat[sSFR]<co.sSFR_THRESHOLD, sSFR] = co.sSFR_QUENCHED
    return cat


def plot_bpt(Catalogue, figsize=(12.0,8.0), label_fontsize=16, tick_labelsize=14, 
             legendmarkerscale=5, name=None, show=False):

    """
    Plot the BPT diagram from the given Catalogue.

    Parameters
    ----------
    Catalogue : pandas.DataFrame
        DataFrame containing at least the columns 'log_NII_Ha', 'log_OIII_Hb', and 'is_AGN'.
    fig_size : tuple, optional
        Figure size (default is (12, 8)).
    label_fontsize : int, optional
        Font size for axis labels (default is 16).
    tick_labelsize : int, optional
        Font size for tick labels (default is 14).
    pdf_filename : str or None, optional
        If provided, the figure is saved as a PDF with this filename. Default is None.

    Returns
    -------
    None
    """
    # Filter out rows with missing values.
    mask = Catalogue['log_NII_Ha'].notnull() & Catalogue['log_OIII_Hb'].notnull()
    df_plot = Catalogue[mask]

    # Separate AGN and non-AGN galaxies.
    agn = df_plot[df_plot['is_AGN']]
    non_agn = df_plot[~df_plot['is_AGN']]

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=figsize)

    # Plot non-AGN galaxies in blue and AGN in red.
    ax.scatter(non_agn['log_NII_Ha'], non_agn['log_OIII_Hb'], 
               color='blue', label='Non-AGN', alpha=0.7, s=0.1)
    ax.scatter(agn['log_NII_Ha'], agn['log_OIII_Hb'], 
               color='red', label='AGN', alpha=0.7, s=0.1)

    # Set axis labels using LaTeX formatting.
    ax.set_xlabel(r'$\log([\mathrm{NII}]/H\alpha)$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\log([\mathrm{OIII}]/H\beta)$', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_labelsize)

    # Add legend and grid.
    ax.legend(markerscale=legendmarkerscale, fontsize=label_fontsize-2, loc='best')
    ax.grid(True)

    # Save the figure if pdf_filename is provided.
    if name:
        fig.savefig(co.FIGURES_PATH + name + '.pdf', format='pdf', bbox_inches='tight')

    if co.SHOW:
        plt.show()



def load_SDSS():
#* --------------------------------------------------------------------------------
#* Querying the SDSS database to retrieve sSFR and Zoo types.
#* We use astroquery.sdss to send SQL queries directly to the SDSS database.
#* --------------------------------------------------------------------------------

    if co.VERBOSE:
        print("Querying the SDSS database...")

    query = f"""
    SELECT 
        s.specObjID,
        s.z,
        p.petroMag_r,
        p.objID,
        g.sfr_tot_p50, g.specsfr_tot_p50, g.lgm_tot_p50,
        l.h_alpha_eqw, l.h_beta_eqw, l.oiii_5007_eqw, l.nii_6584_eqw,
        l.h_alpha_flux, l.h_beta_flux, l.oiii_5007_flux, l.nii_6584_flux,
        z. p_el_debiased AS p_E,
        z. p_cs_debiased     AS p_S
    FROM SpecObj AS s
        JOIN PhotoObj AS p ON s.bestObjID = p.objID
        JOIN galSpecExtra as g ON s.specObjID = g.specObjID
        JOIN galSpecLine as l ON s.specObjID = l.specObjID
        JOIN zooSpec AS z ON s.specObjID = z.specObjID
    WHERE s.z BETWEEN 0.005 AND 0.0452
        AND (p.petroMag_r - p.extinction_r <= 17.77)
        AND s.class = 'GALAXY'
        AND g.lgm_tot_p50 > -1000
    """
    report.append_json('SDSS_query', query,build=True)

    result = SDSS.query_sql(query, data_release=co.DATA_RELEASE)
    #* Execute the query via astroquery
    if result is None or len(result) == 0:
        raise RuntimeError("No data retrieved. Check your query and data release.")


    SDSS_withAGN_df = result.to_pandas()
    report.append_json('SDSS_Gals_with_AGN', len(SDSS_withAGN_df),build=True)
   
    #* Rename columns for consistency
    SDSS_withAGN_df.rename(columns={
        'petroMag_r': 'r_obs',
        'objID': 'objid',
        'sfr_tot_p50': 'SFR',
        'specsfr_tot_p50': 'sSFR',
        'lgm_tot_p50': 'lgm'
        }, inplace=True)

    SDSS_withAGN_df['sSFR_status'] = ''
    SDSS_withAGN_df = sSFR_floor(SDSS_withAGN_df)

    SDSS_withAGN_df['log_NII_Ha'] = mu.safe_log_ratio(SDSS_withAGN_df['nii_6584_flux'], 
                                                      SDSS_withAGN_df['h_alpha_flux'])
    SDSS_withAGN_df['log_OIII_Hb'] = mu.safe_log_ratio(SDSS_withAGN_df['oiii_5007_flux'], 
                                                       SDSS_withAGN_df['h_beta_flux'])

    SDSS_withAGN_df['is_AGN'] = SDSS_withAGN_df.apply(au.classify_agn, axis=1)

    if co.VERBOSE:
        print("   Creating BPT diagram")
    plot_bpt(SDSS_withAGN_df, name="BPT_diagram")

    SDSS_df = remove_AGN(SDSS_withAGN_df)

    if co.VERBOSE:
        print(f"   {len(SDSS_df)} galaxies left after removing AGN")

    report.append_json('SDSS_Gals_AGN', len(SDSS_df),build=True)

    if co.VERBOSE:
        print(f"{len(SDSS_df)} non-AGN galaxies retrieved.")

    return(SDSS_df)
