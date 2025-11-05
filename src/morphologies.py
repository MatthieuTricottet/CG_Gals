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
            report.append_json(f'{cat}_N_{morph}', n_morph, build=True)
            if co.VERBOSE:
                print(f"   {morph}: {n_morph} galaxies")
        

    return sample

def stats(sample):
    """ 
    Perform statistical analysis on the morphological classification of galaxies.
    Parameters
    ----------
    sample : dict
        Dictionary where keys are sample names and values are pandas DataFrames
        containing galaxy data with 'morphology' column.    
    """
    # Morphological fractions
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]+['SDSS']:
        df = sample[cat]
        n_total = len(df)
        for morph in co.Morphologies:  
            n_morph = len(df[df['morphology'] == morph])
            report.append_json(f'{cat}_N_{morph}', n_morph)  
            frac_morph = n_morph / n_total
            report.append_json(f'{cat}_fraction_{morph}_pc', f'{(100*frac_morph):.1f}')  
            if co.VERBOSE:
                print(f"   {cat} - {morph}: {n_morph} galaxies ({(100*frac_morph):.1f}%)")      
    # Statistical tests between CG and Control samples
    for control_name in co.CONTROL.keys():  
        control_cat = control_name + co.GASUFF
        CG_cat = 'CG4' + co.GASUFF
        for morph in ['Elliptical', 'Spiral']:
            # Create contingency table
            n_CG_morph = len(sample[CG_cat][sample[CG_cat]['morphology'] == morph])
            n_CG_non_morph = len(sample[CG_cat]) - n_CG_morph
            n_control_morph = len(sample[control_cat][sample[control_cat]['morphology'] == morph])
            n_control_non_morph = len(sample[control_cat]) - n_control_morph
            contingency_table = np.array([[n_CG_morph, n_CG_non_morph],
                                          [n_control_morph, n_control_non_morph]])
            # Perform Barnard's exact test
            res_barnard = barnard_exact(contingency_table, alternative='two-sided')
            pval = res_barnard.pvalue
            report.append_json(f'pval_{control_name}_{morph}_vs_CG_pc', gu.numformat(pval, prec=1))
            if co.VERBOSE:
                print(f"   p-value for {morph} in {control_name} vs CG: {pval:.3e}")


    
def morph_sSFR(sample):
    """
    Compare morphological and sSFR classifications between compact group galaxies and control samples.

    Parameters
    ----------
    CG : pandas.DataFrame
        DataFrame containing compact group galaxy data.
    Controls : dict
        Dictionary where keys are control sample names and values are pandas DataFrames
        containing control galaxy data.

    Returns
    -------
    dict
        
    """
    status = co.sSFR_status[2] # 'Starforming'

    CG = sample['CG4_Gals']
    CG = CG.loc[CG['morphology'] != co.Morphologies[2]]  # Remove 'Uncertain' morphologies
    CG_m0_sSFR2 = len(CG[(CG['morphology'] == co.Morphologies[0]) & (CG['sSFR_status'] == status)])
    CG_m1_sSFR2 = len(CG[(CG['morphology'] == co.Morphologies[1]) & (CG['sSFR_status'] == status)])
    CG_NoU_sSFR = CG_m0_sSFR2 + CG_m1_sSFR2
    report.append_json(f'CG_Nb_{co.Morphologies[0]}_{status}', CG_m0_sSFR2)
    report.append_json(f'CG_fracpc_{co.Morphologies[0]}_{status}', gu.numformat(100*CG_m0_sSFR2/CG_NoU_sSFR, prec=3))
    report.append_json(f'CG_Nb_{co.Morphologies[1]}_{status}', CG_m1_sSFR2)
    report.append_json(f'CG_fracpc_{co.Morphologies[1]}_{status}', gu.numformat(100*CG_m1_sSFR2/CG_NoU_sSFR, prec=3))

    Controls = {name: sample[name+co.GASUFF] for name in co.CONTROL}

    report.append_json(f'Morph_sSFR_test', 'Barnard two-sided exact test')
    for name, control in Controls.items():
        control = control.loc[control['morphology'] != co.Morphologies[2]]  # Remove 'Uncertain' morphologies
        
        Control_m0_sSFR2 = len(control[(control['morphology'] == co.Morphologies[0]) & (control['sSFR_status'] == status)])
        Control_m1_sSFR2 = len(control[(control['morphology'] == co.Morphologies[1]) & (control['sSFR_status'] == status)])
        Control_NoU_sSFR = Control_m0_sSFR2 + Control_m1_sSFR2
        report.append_json(f'{name}_Nb_{co.Morphologies[0]}_{status}', Control_m0_sSFR2)
        report.append_json(f'{name}_fracpc_{co.Morphologies[0]}_{status}', gu.numformat(100*Control_m0_sSFR2/Control_NoU_sSFR, prec=3))
        report.append_json(f'{name}_Nb_{co.Morphologies[1]}_{status}', Control_m1_sSFR2)
        report.append_json(f'{name}_fracpc_{co.Morphologies[1]}_{status}', gu.numformat(100*Control_m1_sSFR2/Control_NoU_sSFR, prec=3))


        table = [  [CG_m0_sSFR2, CG_m1_sSFR2],
                   [Control_m0_sSFR2, Control_m1_sSFR2]
                ]
        
        res_barnard = barnard_exact(table, alternative='two-sided')
        pval = res_barnard.pvalue
        report.append_json(f'pval_{name}_Starforming_vs_CG_pc', gu.numformat(pval, prec=2))
        if co.VERBOSE:
            print(name)
            print(table)
            print(f"   p-value for Starforming in {name} vs CG: {pval:.3e}")

    # for name, control in Controls.items():
    #     if co.VERBOSE:
    #         print(f"Comparing CG4 with {name}")
    #     # Morphology vs sSFR contingency table
    #     for morph in co.Morphologies:
    #         for status in co.sSFR_status:
    #             n_CG = len(CG[(CG['morphology'] == morph) & (CG['sSFR_status'] == status)])
    #             n_control = len(control[(control['morphology'] == morph) & (control['sSFR_status'] == status)])
    #             report.append_json(f'CG4_vs_{name}_N_{morph}_{status}', n_CG, build=True)
    #             report.append_json(f'Control4B_vs_{name}_N_{morph}_{status}', n_control, build=True)
    #             if co.VERBOSE:
    #                 print(f"   {morph} - {status}: CG4: {n_CG}, {name}: {n_control}")
    