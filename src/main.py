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
import generate_report as report

#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
import data_loader as dl
import generate_report as report
import sSFR
import morphologies as morph



print("Done")

# endregion

def load_data_build():
        
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
    SDSS = dl.load_SDSS()
    if co.VERBOSE:
        print(f"   {len(SDSS)} galaxies loaded from the SDSS in our redshift and magnitude range")
        print("\n")
    sample['SDSS'] = SDSS

    if co.VERBOSE:
        print("Enriching samples with Zoo morphologies from SDSS")  
    morphologies = ['p_E', 'p_S'] 
    for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]:
        sample[cat] = sample[cat].drop(columns=morphologies, 
                                       errors='ignore')
        sample[cat]['objid'] = sample[cat]['objid'].astype('int64')
        SDSS['objid']        = SDSS['objid'].astype('int64')
        sample[cat] = pd.merge(sample[cat], SDSS[['objid']+morphologies], 
                               how='left', on='objid')

    return sample

def sSFR_properties(sample): 

    sample, non_quenched, fit_results, f_interp = sSFR.compute_status(sample)
    report.append_json('sSFR_interp', f_interp, build=True)
 
    for status in co.sSFR_status:
        for cat in [name+co.GASUFF for name in co.SAMPLE.keys()]+['SDSS']:
            report.append_json(f'{cat}_N{status}', len(sample[cat][sample[cat]['sSFR_status'] == status]),build=True)
            report.append_json(f'{cat}_N{status}_pc', f'{(100*len(sample[cat][sample[cat]['sSFR_status'] == status])/len(sample[cat])):.1f}',build=True)
        

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
        report.append_json('pval_'+name,gu.numformat(pval,prec=1),build=True)

    sSFR.BGGs_analysis(sample)

def morph_properties(sample):

    morph.stats(sample)
    morph.morph_sSFR(sample)
    morph.BGGs_analysis(sample)
    
def correlations_by_morph(sample):
    if co.VERBOSE:
        print("Analyzing correlations by morphology...")

def main():
    
    report.initialise_json()

    if co.VERBOSE:
        print("Loading data")

    if co.REBUILD_SAMPLE:
        report.initialise_json(build=True)
        sample = load_data_build()
        if co.VERBOSE:
            print("Classifying morphologies")
        sample = morph.classify_all_samples(sample)

        if co.VERBOSE:
            print("Calculating sSFR properties")
        sSFR_properties(sample)


        report.finalize_json(build=True)
        if co.VERBOSE:
            print("Saving processed samples to disk")
        with open(co.DATA_PATH + co.PROCESS_SAMPLES, "wb") as file:
            pkl.dump(sample, file)

        report.finalize_json(build=True)

    else:
        if co.VERBOSE:
            print("Loading processed samples from disk")
        with open(co.DATA_PATH + co.PROCESS_SAMPLES, "rb") as file:
            sample = pkl.load(file)


    morph_properties(sample)

    correlations_by_morph(sample)

    report.finalize_json()
    report.generate_report()


if __name__ == "__main__":
    main()