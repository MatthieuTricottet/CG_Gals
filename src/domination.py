# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from scipy.stats import fisher_exact, barnard_exact
import pickle as pkl


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
#* Project modules imports
#* --------------------------------------------------------------------------------
import sSFR
import generate_report as report


#* --------------------------------------------------------------------------------
#* Global variables
#* --------------------------------------------------------------------------------
import config as co

#* --------------------------------------------------------------------------------
#* Project data
#* --------------------------------------------------------------------------------

with open(co.DATA_PATH + co.PROCESS_SAMPLES, "rb") as file:
            sample = pkl.load(file)



# endregion

def assess_dom(sample):
    """ Assess domination of the groups in the sample.

        Parameters
        ----------
        sample : dict
            Dictionary containing the samples of galaxies and groups

        Returns
        -------
        sample : dict
            Updated dictionary containing the samples of galaxies and groups with domination information
    """

    for samp in co.SAMPLE:
        Gals_key = samp + co.GASUFF
        Groups_key = samp + co.GRSUFF

        Gals = sample[Gals_key]
        Groups = sample[Groups_key]

        # Create a new column 'is_dominated' in Groups
        Groups['is_dominated'] = Groups['FracLumBGG'] >= co.DOMINATIION_CRITERIA

        # Merge the domination info back to Gals
        Gals = Gals.merge(Groups[['Group', 'is_dominated']], on='Group', how='left')

        sample[Gals_key] = Gals
        sample[Groups_key] = Groups

    return sample

def stat_per_dom(sample):
    """ Compute statistics per domination status.

        Parameters
        ----------
        sample : dict
            Dictionary containing the samples of galaxies and groups

        Returns
        -------
        stats : dict
            Dictionary containing the statistics per domination status
    """

    stats = {}

    for samp in co.SAMPLE:
        Gals_key = samp + co.GASUFF
        Groups_key = samp + co.GRSUFF

        Gals = sample[Gals_key]
        Groups = sample[Groups_key]

        stats[samp] = {}

        for dom_status in [True, False]:
            dom_label = 'Dominated' if dom_status else 'Non-Dominated'

            # Groups statistics
            groups_subset = Groups[Groups['is_dominated'] == dom_status]
            stats[samp][dom_label] = {
                'Mean_FracLumBGG': groups_subset['FracLumBGG'].mean()
            }

            # Galaxies statistics
            gals_subset = Gals[Gals['is_dominated'] == dom_status]
            stats[samp][dom_label].update({
                'Mean_sSFR': gals_subset['sSFR'].mean(),
                'Mean_M_r': gals_subset['M_r'].mean(),
                'Mean_lgm': gals_subset['lgm'].mean(),
                'sSFR_status_counts': gals_subset['sSFR_status'].value_counts().to_dict(),
                'morphology_counts': gals_subset['morphology'].value_counts().to_dict(),
                # 'Radius_Bary_arcmin', 'Offset_Bary', 'V_BGG', 'V_moy', 'Vdisp',
    #    'Voffset', 'size_Group_Bary_kpc', 'M_group', 'M_virial',
    #    'M_virial_over_L', 't_cr', 'BGG_SFRcategory', 'all_SFR', 'Prop_M_Sat',
    #    'Prop_M_Tot', 'Prop_G_Sat', 'Prop_G_Tot', 'Prop_Q_Sat', 'Prop_Q_Tot',
    #    'dom', 'Misfit_Bary', 'Vmisfit', 'lMass_200', 'r_200_kpc'
            })

    return stats