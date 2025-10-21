# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from scipy.stats import fisher_exact, barnard_exact


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
#* Project functions imports
#* --------------------------------------------------------------------------------


# endregion


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a morphological class to each galaxy based on debiased Galaxy Zoo vote fractions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame must contain the following columns:
        - 'p_E' : float  
            Debiased vote fraction for the elliptical classification.
        - 'p_S' : float  
            Debiased vote fraction for the spiral classification.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame augmented with:
        - 'morphology' : str  
            Assigned morphological class, one of:
            'Elliptical', 'Spiral' or 'Uncertain'.
    """
    conds = [
        df['p_E'] > 0.5,
        (df['p_S'] > 0.5)
    ]
    choices = ['Elliptical', 'Spiral']

    df['morphology'] = np.select(conds, choices, default='Uncertain')
    return df

def classify_all_samples(sample: dict) -> dict:
    """
    Apply morphological classification to all galaxy samples in the provided dictionary.

    Parameters
    ----------
    sample : dict
        Dictionary where keys are sample names and values are pandas DataFrames
        containing galaxy data with 'p_E' and 'p_S' columns.

    Returns
    -------
    dict
        The input dictionary with each DataFrame augmented with a 'morphology' column.
    """
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]+["SDSS"]:
        df = sample[cat]
        if co.VERBOSE:
            print(f".  Sample: {cat}")
        df = classify(df)
        for morph in ['Elliptical', 'Spiral', 'Uncertain']:
            n_morph = len(df[df['morphology'] == morph])
            report.append_json(f'{cat}_N{morph}', n_morph, build=True)
            if co.VERBOSE:
                print(f"   {morph}: {n_morph} galaxies")
        

    return sample

def stats(CG,Control):
    """
    Perform statistical analysis on the morphological classification of galaxies.

    Parameters
    ----------
    CG : pandas.DataFrame
        DataFrame containing the morphological classification of galaxies.
    Control : pandas.DataFrame
        DataFrame containing the control sample of galaxies.

    Returns
    -------
    dict
        Dictionary containing the results of the statistical analysis.
    """
    results = {}
    Nb_S_Control = len(Control[Control['morphology'] == 'Spiral'])
    results['Nb_S_Control'] = Nb_S_Control
    Nb_E_Control = len(Control[Control['morphology'] == 'Elliptical'])
    results['Nb_E_Control'] = Nb_E_Control
    Nb_S_CG = len(CG[CG['morphology'] == 'Spiral'])
    results['Nb_S_CG'] = Nb_S_CG
    Nb_E_CG = len(CG[CG['morphology'] == 'Elliptical'])
    results['Nb_E_CG'] = Nb_E_CG
    fraction_S_Control = Nb_S_Control / (Nb_S_Control + Nb_E_Control)
    results['fraction_S_Control_pc'] = fraction_S_Control
    fraction_E_Control = Nb_E_Control / (Nb_S_Control + Nb_E_Control)
    results['fraction_E_Control_pc'] = fraction_E_Control
    fraction_S_CG = Nb_S_CG / (Nb_S_CG + Nb_E_CG)
    results['fraction_S_CG_pc'] = fraction_S_CG
    fraction_E_CG = Nb_E_CG / (Nb_S_CG + Nb_E_CG)
    results['fraction_E_CG_pc'] = fraction_E_CG

    table = [[Nb_S_Control, Nb_E_Control],
            [Nb_S_CG, Nb_E_CG]]
    res_fisher = fisher_exact(table, alternative='two-sided')
    pval_morphologies_CGvsControl = res_fisher.pvalue
    results['pval_morphologies_CGvsControl_pc'] = pval_morphologies_CGvsControl

    return results
    
#    def morph_sSFR(CG,Control):
#     """
#     Perform statistical analysis on the specific star formation rate (sSFR) of galaxies based on their morphological classification.

#     Parameters
#     ----------
#     CG : pandas.DataFrame
#         DataFrame containing the morphological classification and sSFR of galaxies.
#     Control : pandas.DataFrame
#         DataFrame containing the control sample of galaxies.

#     Returns
#     -------
#     dict
#         Dictionary containing the results of the statistical analysis.
#     """
#     results = {}
    
#     # sSFR for Control sample
#     sSFR_Control = Control['sSFR'].values
#     results['sSFR_Control_mean'] = np.mean(sSFR_Control)
#     results['sSFR_Control_std'] = np.std(sSFR_Control)

#     # sSFR for CG sample
#     sSFR_CG = CG['sSFR'].values
#     results['sSFR_CG_mean'] = np.mean(sSFR_CG)
#     results['sSFR_CG_std'] = np.std(sSFR_CG)

#     # Statistical test between CG and Control samples
#     res_barnard = barnard_exact(sSFR_CG, sSFR_Control, alternative='two-sided')
#     pval_sSFR_CGvsControl = res_barnard.pvalue
#     results['pval_sSFR_CGvsControl_pc'] = pval_sSFR_CGvsControl

#     return results