# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from scipy.stats import fisher_exact


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
            'Elliptical', 'Spiral', 'Uncertain', or the project missing-vote
            label. ``Uncertain`` means finite Galaxy Zoo votes that do not
            cross either threshold; missing vote fractions are not folded into
            that class.
    """
    p_e = pd.to_numeric(df.get('p_E', pd.Series(np.nan, index=df.index)), errors='coerce')
    p_s = pd.to_numeric(df.get('p_S', pd.Series(np.nan, index=df.index)), errors='coerce')
    finite_votes = p_e.notna() & p_s.notna()

    morphology = pd.Series(co.NoMorphology_LABEL, index=df.index, dtype=object)
    morphology.loc[finite_votes] = co.Morphologies[-1]
    morphology.loc[finite_votes & (p_e > 0.5)] = co.Morphologies[0]
    morphology.loc[finite_votes & ~(p_e > 0.5) & (p_s > 0.5)] = co.Morphologies[1]

    df['morphology'] = morphology
    return df

def classify_all_samples(sample: dict, record_build_counts: bool = True) -> dict:
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
        if cat not in sample:
            continue
        df = sample[cat]
        if co.VERBOSE:
            print(f".  Sample: {cat}")
        df = classify(df)
        sample[cat] = df
        if record_build_counts:
            for morph in [*co.Morphologies, co.NoMorphology_LABEL]:
                n_morph = int((df['morphology'] == morph).sum())
                report.append_json(f'{cat}_N_{morph}', n_morph, build=True)
                if co.VERBOSE:
                    print(f"   {morph}: {n_morph} galaxies")
            p_e = pd.to_numeric(df.get('p_E', pd.Series(np.nan, index=df.index)), errors='coerce')
            p_s = pd.to_numeric(df.get('p_S', pd.Series(np.nan, index=df.index)), errors='coerce')
            finite_votes = p_e.notna() & p_s.notna()
            report.append_json(f'{cat}_N_GZFinite', int(finite_votes.sum()), build=True)
            report.append_json(f'{cat}_N_GZMissing', int((~finite_votes).sum()), build=True)
            report.append_json(f'{cat}_N_p_E_finite', int(p_e.notna().sum()), build=True)
            report.append_json(f'{cat}_N_p_S_finite', int(p_s.notna().sum()), build=True)


    # ADD MORPHOLOGICAL FRACTIONS TO GROUPS
        
    # for cat in [name+co.GRPSUFF for name in co.SAMPLE.keys()]:
    #     # Add to groups the fraction of spiral and elliptical galaxies, either considering or eliminating uncertain morphologies




    #     def loc_agg(df):

    #     df = sample[cat]
    #     df['S_frac_U'] = 

    return sample

def add_morphology_fractions_to_groups(sample: dict) -> dict:
    """
    Add morphological fractions of member galaxies to each group in the provided samples.

    Parameters
    ----------
    sample : dict
        Dictionary where keys are sample names and values are pandas DataFrames
        containing galaxy and group data.

    Returns
    -------
    dict
        The input dictionary with each group DataFrame augmented with morphological fraction columns.
    """
    for samp in co.SAMPLE.keys():
        Gals_key = samp + co.GASUFF
        Groups_key = samp + co.GRSUFF

        Gals = sample[Gals_key]
        Groups = sample[Groups_key]

        
        # Calculate morphological fractions for each group: with or excluding 'Uncertain', with or excluding the BGG

        for exclude_uncertain in [True, False]:
            for exclude_BGG in [True, False]:
                loc_Gals = Gals[Gals['morphology'].isin(co.Morphologies)].copy()
                suffix = '_frac'
                if exclude_uncertain:
                    loc_Gals = loc_Gals[loc_Gals['morphology'].isin(co.Morphologies[:2])]
                    suffix += '_NoU'
                if exclude_BGG:
                    loc_Gals = loc_Gals[loc_Gals['rank_M'] != 1]
                    suffix += '_NoBGG'

                sizes = loc_Gals['Group'].value_counts()
                composition = (
                    loc_Gals.groupby(['Group', 'morphology'])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=['Elliptical', 'Spiral'], fill_value=0)
                )
                proportions = composition.div(sizes, axis=0)
                renamed = proportions[['Elliptical', 'Spiral']].rename(
                    columns={'Elliptical': f'E{suffix}', 'Spiral': f'S{suffix}'}
                )
                Groups = Groups.drop(columns=renamed.columns, errors='ignore')
                Groups = Groups.merge(renamed, left_on='Group', right_index=True, how='left')
                    
        sample[Groups_key] = Groups


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
        for morph in [*co.Morphologies, co.NoMorphology_LABEL]:
            n_morph = len(df[df['morphology'] == morph])
            report.append_json(f'{cat}_N_{morph}', n_morph)  
            frac_morph = n_morph / n_total
            report.append_json(f'{cat}_fraction_{morph}_pc', f'{(100*frac_morph):.1f}')  
            if co.VERBOSE:
                print(f"   {cat} - {morph}: {n_morph} galaxies ({(100*frac_morph):.1f}%)")      
        n_usable = int(df['morphology'].isin(co.Morphologies[:2]).sum())
        n_excluded = int(n_total - n_usable)
        report.append_json(f'{cat}_N_GZUsable', n_usable)
        report.append_json(f'{cat}_N_GZExcludedBinary', n_excluded)
        report.append_json(
            f'{cat}_fraction_GZExcludedBinary_pc',
            f'{(100*n_excluded/n_total):.1f}' if n_total else 'nan',
        )
    # Statistical tests between CG and Control samples
    for control_name in co.CONTROL.keys():  
        control_cat = control_name + co.GASUFF
        CG_cat = 'CG4' + co.GASUFF
        CG_usable = sample[CG_cat][
            sample[CG_cat]['morphology'].isin(co.Morphologies[:2])
        ]
        control_usable = sample[control_cat][
            sample[control_cat]['morphology'].isin(co.Morphologies[:2])
        ]
        contingency_table = np.array(
            [
                [
                    int(CG_usable['morphology'].eq('Elliptical').sum()),
                    int(CG_usable['morphology'].eq('Spiral').sum()),
                ],
                [
                    int(control_usable['morphology'].eq('Elliptical').sum()),
                    int(control_usable['morphology'].eq('Spiral').sum()),
                ],
            ]
        )
        pval = fisher_exact(contingency_table, alternative='two-sided').pvalue
        report.append_json(
            f'pval_{control_name}_Elliptical_vs_CG_pc', gu.pvalue_latex(pval)
        )
        if co.VERBOSE:
            print(
                f"   Fisher p-value for the E/Sp mix in {control_name} vs CG: "
                f"{pval:.3e}"
            )

    
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
    status = co.sSFR_status[-1] # 'Starforming'

    CG = sample['CG4_Gals']
    CG = CG.loc[CG['morphology'].isin(co.Morphologies[:2])]
    CG_m0_sSFR2 = len(CG[(CG['morphology'] == co.Morphologies[0]) & (CG['sSFR_status'] == status)])
    CG_m1_sSFR2 = len(CG[(CG['morphology'] == co.Morphologies[1]) & (CG['sSFR_status'] == status)])
    CG_NoU_sSFR = CG_m0_sSFR2 + CG_m1_sSFR2
    report.append_json(f'CG_Nb_{co.Morphologies[0]}_{status}', CG_m0_sSFR2)
    report.append_json(f'CG_fracpc_{co.Morphologies[0]}_{status}', gu.numformat(100*CG_m0_sSFR2/CG_NoU_sSFR, prec=3))
    report.append_json(f'CG_Nb_{co.Morphologies[1]}_{status}', CG_m1_sSFR2)
    report.append_json(f'CG_fracpc_{co.Morphologies[1]}_{status}', gu.numformat(100*CG_m1_sSFR2/CG_NoU_sSFR, prec=3))

    Controls = {name: sample[name+co.GASUFF] for name in co.CONTROL}

    report.append_json('Morph_sSFR_test', 'two-sided Fisher exact test')
    for name, control in Controls.items():
        control = control.loc[control['morphology'].isin(co.Morphologies[:2])]
        
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
        
        pval = fisher_exact(table, alternative='two-sided').pvalue
        report.append_json(f'pval_{name}_Starforming_vs_CG_pc', gu.pvalue_latex(pval))
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
    

def BGGs_analysis(sample):
    """
    Analyze the morphological and sSFR properties of Brightest Group Galaxies (BGGs) in compact groups and control samples.

    Parameters
    ----------
    sample : dict
        Dictionary where keys are sample names and values are pandas DataFrames
        containing galaxy data with 'morphology', 'sSFR_status', and 'rank_M' columns.
    """
    if co.VERBOSE:
        print("Analyzing BGGs morphological properties...")

    report.append_json('BGG_morph_tests', 'two-sided Fisher exact test')

    CG4 = sample['CG4'+co.GASUFF]
    BGGs_CG4 = CG4[CG4['rank_M'] == 1]
    BGGs_CG4_finite = BGGs_CG4[BGGs_CG4['morphology'].isin(co.Morphologies)]
    BGGs_CG4_usable = BGGs_CG4[
        BGGs_CG4['morphology'].isin(co.Morphologies[:2])
    ]
    total = len(BGGs_CG4_usable)
    if co.VERBOSE:
        print(f"Sample: CG4 - Total BGGs with usable E/Sp classifications: {total}")
    report.append_json(f'CG4_BGG_N_{co.NoMorphology_LABEL}', int((BGGs_CG4['morphology'] == co.NoMorphology_LABEL).sum()))
    report.append_json('CG4_BGG_N_GZFinite', len(BGGs_CG4_finite))
    report.append_json('CG4_BGG_N_GZUsable', total)
    for morph in co.Morphologies:
        n_BGGs = len(BGGs_CG4_finite[BGGs_CG4_finite['morphology'] == morph])
        report.append_json(f'CG4_BGG_N_{morph}', n_BGGs)
        if morph in co.Morphologies[:2]:
            report.append_json(
                f'CG4_BGG_fracpc_{morph}',
                f"{100*n_BGGs/total:.1f}" if total else "nan",
            )
        if co.VERBOSE:
            frac = n_BGGs / total if total else np.nan
            print(f".  {morph}: {n_BGGs} / {total} = {frac:.1f}")
    

    for cat in co.CONTROL.keys():
        name = cat+co.GASUFF
        df = sample[name]
        BGGs = df[df['rank_M'] == 1]
        BGGs_finite = BGGs[BGGs['morphology'].isin(co.Morphologies)]
        BGGs_usable = BGGs[BGGs['morphology'].isin(co.Morphologies[:2])]
        total = len(BGGs_usable)
        if co.VERBOSE:
            print(f"Sample: {cat} - Total BGGs with usable E/Sp classifications: {total}")
        report.append_json(f'{cat}_BGG_N_{co.NoMorphology_LABEL}', int((BGGs['morphology'] == co.NoMorphology_LABEL).sum()))
        report.append_json(f'{cat}_BGG_N_GZFinite', len(BGGs_finite))
        report.append_json(f'{cat}_BGG_N_GZUsable', total)
        for morph in co.Morphologies:
            n_BGGs = len(BGGs_finite[BGGs_finite['morphology'] == morph])
            report.append_json(f'{cat}_BGG_N_{morph}', n_BGGs)
            if morph in co.Morphologies[:2]:
                report.append_json(
                    f'{cat}_BGG_fracpc_{morph}',
                    f"{100*n_BGGs/total:.1f}" if total else "nan",
                )
            if co.VERBOSE:
                print(f"   {morph}: {n_BGGs} BGGs")
        
        matrix = [[len(BGGs_CG4[BGGs_CG4['morphology'] == co.Morphologies[0]]), len(BGGs_CG4[BGGs_CG4['morphology'] == co.Morphologies[1]])],
                  [len(BGGs[BGGs['morphology'] == co.Morphologies[0]]), len(BGGs[BGGs['morphology'] == co.Morphologies[1]])]]
        res_fisher = fisher_exact(matrix, alternative='two-sided')
        report.append_json(f'BGG_morph_pvalue_{cat}_vs_CG4', f'{res_fisher.pvalue:.2f}')
        if co.VERBOSE:
            print("Exact test p-values of proportion of morphologies being different between CG_4 BGGs and the control sample BGGs:")
            print(f"   Fisher: {res_fisher.pvalue:.1e}")
            if res_fisher.pvalue < 0.05:
                print("   Reject null hypothesis: the proportion of morphologies in BGGs is different between CG_4 and the control sample")
            else:
                print("   Fail to reject null hypothesis: the proportion of morphologies in BGGs is not different between CG_4 and control sample")    
    
