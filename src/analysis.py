# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
try:
    from astroquery.sdss import SDSS
except ModuleNotFoundError:  # pragma: no cover - optional dependency for other analyses
    SDSS = None

try:
    from scipy.stats import multivariate_normal
except ModuleNotFoundError:  # pragma: no cover - optional dependency for other analyses
    multivariate_normal = None

try:
    import scipy.interpolate as interp
except ModuleNotFoundError:  # pragma: no cover - optional dependency for other analyses
    interp = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.lines as mlines  # for legend proxies
    import matplotlib.ticker as ticker
except ModuleNotFoundError:  # pragma: no cover - optional dependency for other analyses
    plt = ListedColormap = mlines = ticker = None

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency for other analyses
    sns = None

import time 
import re 
from io import StringIO
import pickle as pkl


#* --------------------------------------------------------------------------------
#* Personal librairies imports
#* --------------------------------------------------------------------------------
import sys, os
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
try:
    from utils import astro_utils as au
    from utils import maths_utils as mu
    from utils import stats_utils as su
    from utils import graphics_utils as gu
    from utils import labels_utils as lu
    from utils import pandas_utils as pu
except ModuleNotFoundError:  # pragma: no cover - optional dependencies for other analyses
    au = mu = su = gu = lu = pu = None

#* --------------------------------------------------------------------------------
#* Global variables
#* --------------------------------------------------------------------------------
try:
    import config as co
    import generate_report as report
    import domination as dom
except ModuleNotFoundError:  # pragma: no cover - optional dependencies for other analyses
    co = report = dom = None


#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
try:
    import data_loader as dl
    import generate_report as report
    import sSFR
    import morphologies as morph
except ModuleNotFoundError:  # pragma: no cover - optional dependencies for other analyses
    dl = report = sSFR = morph = None



print("Done")

# endregion

def stats_comp_split(split):
    """ compares two parts of a split sample and computes statistics
        Parameters
        ----------
        split : dict
            Dictionary containing the two parts of the split sample and their names, by gals and groups

        Returns
        -------
        stats : dict
            Dictionary containing the statistics of the two parts of the split sample
    """
    stats = {}

    for part in split.keys():
        part_data = split[part]
        if isinstance(part_data, dict) and 'Gals' in part_data:
            gals_subset = part_data['Gals']
            groups_subset = part_data.get('Groups')
        else:
            gals_subset = part_data
            groups_subset = None

        stats[part] = {
            'sSFR': gals_subset['sSFR'].mean(),
            'M_r': gals_subset['M_r'].mean(),
            'lgm': gals_subset['lgm'].mean(),
            'sSFR_status_counts': gals_subset['sSFR_status'].value_counts().to_dict(),
            'morphology_counts': gals_subset['morphology'].value_counts().to_dict(),
        }

        if 'BGG_SFRcategory' in gals_subset.columns:
            stats[part]['BGG_SFRcategory'] = gals_subset['BGG_SFRcategory'].value_counts().to_dict()

        if groups_subset is not None:
            for key in [
                'Offset_Bary',
                'Vdisp',
                'Voffset',
                'size_Group_Bary_kpc',
                'M_group',
                'M_virial',
                'M_virial_over_L',
                't_cr',
                'Prop_M_Sat',
                'Prop_M_Tot',
                'Prop_G_Sat',
                'Prop_G_Tot',
                'Prop_Q_Sat',
                'Prop_Q_Tot',
                'dom',
                'Misfit_Bary',
                'Vmisfit',
                'lMass_200',
                'r_200_kpc',
            ]:
                stats[part][key] = groups_subset[key].mean()

    return stats


def stats_comp_split_per_BGG(split): # DRAFT!!!!!!!!
    """ compares two parts of a split sample and computes statistics
        Parameters
        ----------
        split : dict
            Dictionary containing the two parts of the split sample and their names, by gals and groups

        Returns
        -------
        stats : dict
            Dictionary containing the statistics of the two parts of the split sample
    """
    stats = {}

    for part in split.keys():
        Gals = split[part]['Gals']
        Groups = split[part]['Groups']
        gals_subset = Gals
        groups_subset = Groups

        stats[part] = {
                'sSFR': gals_subset['sSFR'].median(),
                'M_r': gals_subset['M_r'].median(),
                'lgm': gals_subset['lgm'].median(),
                'sSFR_status_counts': gals_subset['sSFR_status'].value_counts().to_dict(),
                'morphology_counts': gals_subset['morphology'].value_counts().to_dict(),
                'Offset_Bary': groups_subset['Offset_Bary'].median(),
                'Vdisp': groups_subset['Vdisp'].median(),
                'Voffset': groups_subset['Voffset'].median(),
                'size_Group_Bary_kpc' : groups_subset['size_Group_Bary_kpc'].median(), 
                'M_group' : groups_subset['M_group'].median(), 
                'M_virial' : groups_subset['M_virial'].median(),
                'M_virial_over_L' : groups_subset['M_virial_over_L'].median(), 
                't_cr' : groups_subset['t_cr'].median(), 
                'BGG_SFRcategory' : gals_subset['BGG_SFRcategory'].value_counts().to_dict(), 
                'Prop_M_Sat' : groups_subset['Prop_M_Sat'].median(),
                'Prop_M_Tot'    : groups_subset['Prop_M_Tot'].median(), 
                'Prop_G_Sat' : groups_subset['Prop_G_Sat'].median(), 
                'Prop_G_Tot' : groups_subset['Prop_G_Tot'].median(), 
                'Prop_Q_Sat' : groups_subset['Prop_Q_Sat'].median()   , 
                'Prop_Q_Tot' : groups_subset['Prop_Q_Tot'].median(),
                'dom' : groups_subset['dom'].median(), 
                'Misfit_Bary' : groups_subset['Misfit_Bary'].median(), 
                'Vmisfit' : groups_subset['Vmisfit'].median(), 
                'lMass_200' : groups_subset['lMass_200'].median(), 
                'r_200_kpc' : groups_subset['r_200_kpc'].median()
            }

    return stats
