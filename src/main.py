# region Imports

print("Loading libraries...")

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
import generate_report as report
import domination as dom


#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
import data_loader as dl
import generate_report as report
import sSFR
import morphologies as morph
import analysis as anl
import exploration_dom
import exploration_coulours
import exploration_morph
import exploration_ssfr
import exploration_mvt_tcross
import extended_specialness
import luminosity_function
import ssfr_robustness



print("Done")

# endregion

def clean(sample):
    """ Removes spurious groups comming from Lim group 3688

        Parameters
        ----------
        sample : dict
            Dictionary containing the samples of galaxies and groups

        Returns
        -------
        sample : dict
            Cleaned dictionary containing the samples of galaxies and groups
    """
    sample['Control4B_Gals'] = sample['Control4B_Gals'][sample['Control4B_Gals']['Group'] != 3688]
    sample['Control4B_Groups'] = sample['Control4B_Groups'][sample['Control4B_Groups']['Group'] != 3688]

    sample['Control4C_Gals'] = sample['Control4C_Gals'][sample['Control4C_Gals']['Group'] != 3688]
    sample['Control4C_Groups'] = sample['Control4C_Groups'][sample['Control4C_Groups']['Group'] != 3688]

    return sample


def load_data():
    """ Load or rebuild the data sample.

        Parameters
        ----------
        None

        Returns
        -------
        sample : dict
            Dictionary containing the samples of galaxies and groups
    """
    
    if co.VERBOSE:
        print("Loading data")

    if co.REBUILD_SAMPLE:
        report.initialise_json(build=True)
        sample = load_data_build()
        sample = clean(sample)
        if co.VERBOSE:
            print("Classifying morphologies")
        sample = morph.classify_all_samples(sample)

        sample = morph.add_morphology_fractions_to_groups(sample)

        if co.VERBOSE:
            print("Calculating sSFR properties")
        sample = sSFR_properties(sample)

        report.finalize_json(build=True)
        if co.VERBOSE:
            print("Saving processed samples to disk")
        with open(co.DATA_PATH + co.PROCESS_SAMPLES, "wb") as file:
            pkl.dump(sample, file)

    else:
        if co.VERBOSE:
            print("Loading processed samples from disk")
        with open(co.DATA_PATH + co.PROCESS_SAMPLES, "rb") as file:
            sample = pkl.load(file)
        if "SDSS_withAGN" in sample:
            dl.plot_bpt(sample["SDSS_withAGN"], name="BPT_diagram")

    return sample


def load_data_build():
    """Rebuild the processed catalogue from raw CSV inputs and SDSS enrichments."""
        
    sample = dl.load_previous_samples()
    report.append_json('CG4_Gals_withsplit_N', len(sample['CG4_Gals']), build=True)
    report.append_json('CG4_Groups_withsplit_N', len(sample['CG4_Groups']), build=True)  

    if co.VERBOSE:
        print("\nRemoving split CG galaxies and groups")
    sample['CG4_Gals'],sample['CG4_Groups'] = dl.remove_split_CG(sample['CG4_Gals'],
                                                                 sample['CG4_Groups'])
    report.append_json('CG4_Gals_nonsplit_N', len(sample['CG4_Gals']), build=True)
    report.append_json('CG4_Groups_nonsplit_N', len(sample['CG4_Groups']), build=True)                                                          
    if co.VERBOSE:
        print(f"   {len(sample['CG4_Gals'])} galaxies left in CG4_Gals")
        print(f"   {len(sample['CG4_Groups'])} groups left in CG4_Groups")
        print("\n")

    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]:
        sample[cat] = dl.sSFR_floor(sample[cat])

    if co.VERBOSE:
        print("Loading SDSS data")
    SDSS_withAGN, SDSS  = dl.load_SDSS()
    if co.VERBOSE:
        print(f"   {len(SDSS)} galaxies loaded from the SDSS in our redshift and magnitude range")
        print("\n")
    sample['SDSS'] = SDSS
    sample['SDSS_withAGN'] = SDSS_withAGN

    if co.VERBOSE:
        print("Enriching samples with Zoo morphologies from SDSS")  
    morphologies = ['p_E', 'p_S'] 
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]:
        sample[cat] = sample[cat].drop(columns=morphologies, 
                                       errors='ignore')
        sample[cat]['objid'] = sample[cat]['objid'].astype('int64')
        SDSS_withAGN['objid'] = SDSS_withAGN['objid'].astype('int64')
        sample[cat] = pd.merge(sample[cat], SDSS_withAGN[['objid']+morphologies], 
                               how='left', on='objid')

    sample = dom.assess_dom(sample)
    return sample

def sSFR_properties(sample): 
    """Compute the sSFR-derived quantities, figures and report entries."""

    sample, non_quenched, fit_results, f_interp = sSFR.compute_status(sample)
    report.append_json('sSFR_interp', f_interp, build=True)
    main_sequence = sample['SDSS'][sample['SDSS']['sSFR_status']=='Starforming'][['lgm','sSFR']]

    # Plot main sequence and get coefficients model
    # Save them 
    MS_coeffs = sSFR.plot_main_sequence_models(main_sequence)
    with open(co.DATA_PATH + co.PROCESS_SAMPLES, "wb") as file:
            pkl.dump(sample, file)

    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]:
        sample[cat] = sSFR.add_MS_offset(sample[cat], MS_coeffs)
    
    model = sSFR.fit_ssfr_vs_lgm_poly(main_sequence, order=2)
    sSFR.add_MS_residuals(sample, model, suffix=co.GASUFF, non_sf_value=np.nan)
    sSFR.plot_main_sequence_residuals(sample, figname='main_sequence_residuals')
    significance = sSFR.compare_main_sequence_residuals_bootstrap(sample)
    # Report main sequence residuals significance
    for name, pval in significance.items():
        # report.append_json('pval_MSresiduals_'+name, gu.numformat(pval, prec=1), build=True)
        report.append_json('pval_MSresiduals_'+name, pval, build=True)

 
    for status in co.sSFR_status:
        for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]+['SDSS']:
            report.append_json(f'{cat}_N{status}', len(sample[cat][sample[cat]['sSFR_status'] == status]),build=True)
            report.append_json(f'{cat}_N{status}_pc', f'{(100*len(sample[cat][sample[cat]['sSFR_status'] == status])/len(sample[cat])):.1f}',build=True)
        
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]:
        sSFR.add_excess(sample[cat], f_interp)

    if co.VERBOSE:
        print("   Plotting sSFR gaussian mixture")
    sSFR.plot_density_original_vs_GMMfit(non_quenched[['lgm', 'sSFR']].values, fit_results, name="density_original_vs_GMMfit") 

    if co.VERBOSE:
        print("   Plotting sSFR classification and decision boundary")
    sSFR.plot_classification(non_quenched, sample['SDSS'], fit_results, f_interp, name='sSFR_classification')

    if co.VERBOSE:
        print("   Plotting galaxies sSFR")
    sSFR.plot_galaxies(sample['SDSS'], sample['CG4_Gals'], name='galaxies_sfr')

    if co.VERBOSE:
        print("   Plotting residual distribution")
    sSFR.plot_residual_distribution(non_quenched,f_interp, name="residual_sSFR_histogram", )

    results = sSFR.compare(sample, Verbose=True)
    for name,pval in results.items():
        report.append_json('pval_'+name, gu.pvalue_latex(pval), build=True)

    return(sample)

def morph_properties(sample):
    """Run the morphology analyses that populate the report JSON."""

    morph.stats(sample)
    morph.morph_sSFR(sample)
    morph.BGGs_analysis(sample)


def ssfr_report_properties(sample):
    """Write per-run sSFR comparison results used directly by the paper."""

    results = sSFR.compare(sample, Verbose=True)
    for name, pval in results.items():
        report.append_json('pval_' + name, gu.pvalue_latex(pval))

    sSFR.BGGs_analysis(sample)
    
def correlations_by_morph(sample):
    """Generate the morphology-split correlation products."""

    if co.VERBOSE:
        print("Analyzing correlations by morphology...")

# def dom_properties(sample):


def run_explorations(sample):
    """Execute the factorized exploration modules used by the paper pipeline."""

    if co.VERBOSE:
        print("Running exploration analyses")

    exploration_dom.run(sample)
    exploration_coulours.run(sample)
    exploration_morph.run(sample)
    exploration_ssfr.run(sample)
    exploration_mvt_tcross.run(sample)


def main():
    """Run the configured analysis pipeline and regenerate the paper outputs."""
    
    if co.RENDER_PAPER_ONLY:
        if co.VERBOSE:
            print("Rendering paper from existing JSON outputs")
        report.generate_report()
        return

    report.initialise_json()
    sample = load_data()

    try:
        lf_results = luminosity_function.run_luminosity_function_analysis(sample)
    except Exception as exc:
        lf_results = {
            "status": "failed",
            "reason": "analysis_exception",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        if co.VERBOSE:
            print(f"[luminosity function] failed: {exc}")
    report.append_json("luminosity_function", lf_results)

    ssfr_report_properties(sample)
    morph_properties(sample)

    sSFR.split_by_fertility(sample)
    sSFR.split_by_BGG_fertility(sample)
    sSFR.satellites_split_by_BGG_fertility(sample)

    run_explorations(sample)
    ssfr_robustness.run(sample)
    extended_specialness.run_extended_specialness(sample)
    

    # correlations_by_morph(sample)

    report.finalize_json()
    report.generate_report()


if __name__ == "__main__":
    main()
