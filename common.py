#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:34:10 2021

@author: matt
"""



# Import libraries

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as plticker
import seaborn as sns

from scipy.stats import spearmanr, kstest, wilcoxon, anderson, fisher_exact, \
                        kendalltau, ranksums, pearsonr, ks_2samp, binned_statistic


from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky

from astropy.cosmology import FlatLambdaCDM

import itertools as it

import re
import timeit

import datetime

from num2tex import num2tex
from num2tex import configure as num2tex_configure
num2tex_configure(help_text=False)

from scipy import optimize # for M_200c computation



import pandas as pd

import sys
sys.path.append('/Users/matt/Google Drive/Astrophysics/MattUtils')
import sphutils as su # Matt
import phyutils as pu # Matt


# Globals variable
def glob():
    pd.set_option('display.max_columns', None) #Eases my life...
    pd.set_option('display.min_rows', 30) #Eases my life too...
    
    global datafolder, figfolder, figfolder2
    datafolder = "../Data/"
    figfolder = "../Figures/"
    figfolder2 = "../Figures_paper2/"
    global Tempel, HMCG, Yang, Zoo
    Tempel = "Tempel/"
    HMCG = "HMCG/"
    Yang = "YangDR13/catalogs/"
    Zoo = "Zoo/"
    
    
    global NbGal, NbGalMin, Deltamag
    NbGal = NbGalMin = 4
    Deltamag = 3
    
    global zmin,zgroupmin
    zmin = 0.005
    zgroupmin = 0.00836 # ∆v=1000 km/s = c ∆z / (1+z_group)
    
    global mag_lim
    mag_lim = 17.77
    
    global Mr_Sol, c
    Mr_Sol = 4.68
    c = 299792.458
    
    global G
    G=6.6743e-11

    global cosmo,Om0,H0,h
    # Planck 2015 Cosmological Parameters
    H0 = 67.8  # Hubble constant in km/s/Mpc
    h = H0/100
    Om0 = 0.308  # Matter density parameter
    Tcmb0 = 2.7255  # CMB temperature today in Kelvin (default for Planck)
    Neff = 3.15  # Effective number of neutrino species
    
    # Initialize the FlatLambdaCDM cosmology
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0, Neff=Neff)


    
    global zmax, Rmax 
    
    global sSFRwidth
    sSFRwidth = 0.4
    
    global gal_suff,group_suff
    gal_suff = "_Gals"
    group_suff = "_Groups"
    global sample_list, Gals_list, Groups_list, sample_colour, sample_label, sample_label_short, control_list
    global sample_list_name, control_list_name
    global centre_label, centre_label_short
    global rank_colour, rank_label, sfr_colour, selectorlist, selector_colour, sample_colour_label
    global selectors, quantities
    # sample_list = ["CG4","Control4B","Control4C","Control4CB"]
    sample_list = ["CG4","Control4B","Control4CB","ControlPC4"]
    sample_list_name = {"CG4" : "\CG","Control4B":"\CB","Control4CB":"\CC","ControlPC4":"\PCF"}
    control_list = [samp for samp in sample_list if samp not in ['CG4']]
    control_list_name = {samp : sample_list_name[samp] for samp in control_list}

    selectors = {
        'BGG_SFRcategory'    : None, 
        'dom'                : 'DeltaR12', 
        'Misfit_Bary'        : 'Offset_Bary', 
        'Misfit_Circ'        : 'Offset_Circ', 
        'Vmisfit'            : 'Voffset'
    }
    # quantities = ['DeltaR12', 'FracLumBGG', 'Offset_Bary', 'Offset_Circ', 'Voffset', 'Vdisp', 
                  # 'size_Group_Bary_kpc','size_Group_Circ_kpc']
    quantities = ['DeltaR12', 'FracLumBGG', 'Offset_Bary', 'Voffset', 'Vdisp', 
                  'size_Group_Bary_kpc']


    Gals_list = ["{}{}".format(i,gal_suff) for i in sample_list]
    Groups_list = ["{}{}".format(i,group_suff) for i in sample_list]
    
    sample_colour = {
                     "CG4"       : "blue",
                     "Control4B" : "lime",
                     "Control4CB" : "salmon",
                     # "ControlPC4" : "goldenrod"
                     # "ControlPC4" : "darkorange"
                     "ControlPC4" : "darkgoldenrod"
    }
    selector_colour = {
        'BGG_SFRcategory' : {
                     "Q"    : "red",
                     "M"    : "blue",
                     "G"    : "green"},
        'BGG_Morph' : {
                     "E"    : "red",
                     "S"    : "blue",
                     "S0"    : "green"},
        'dom'             : {
                     "Non dominated" : "blue",
                     "Dominated"     : "red"}, 
        'Misfit_Bary'     : {
                     "BCentered" : "blue",
                     "BMisfit"     : "red"}, 
        'Misfit_Circ'     : {
                     "CCentered" : "blue",
                     "CMisfit"     : "red"}, 
        'Vmisfit'     : {
                     "VCentered" : "blue",
                     "VMisfit"     : "red"}, 
        }
    sample_label = {
                     "CG4"    : "CG$_4$",
                     "Control4B" : "Control$_\mathrm{4\mathrm{B}}$",
                     # "Control4C" : "Control$_{4\mathrm{C}}$",
                     "Control4CB" : "Control$_{4\mathrm{C}}$",
                     "ControlPC4" : "RG$_4$"
    }
    
    sample_label_short = {
                     "CG4"    : "CG$_4$",
                     "Control4B" : "C$_\mathrm{4\mathrm{B}}$",
                     # "Control4C" : "Control$_{4\mathrm{C}}$",
                     "Control4CB" : "C$_{4\mathrm{C}}$",
                     "ControlPC4" : "RG$_4$"
    }
        
    sample_colour_label = {sample_label[cat] : sample_colour[cat] for cat in sample_label.keys()}
    
    rank_colour  = {
                    "BGG" : "darkmagenta",
                    "Sat" : "steelblue"                    
    }
    rank_label  = {
                    "BGG" : "BGG",
                    "Sat" : "Satellites"                    
    }
    centre_label = {
                    "Bary" : "Centroid", 
                    "Circ" : "Circumscribed circle"
        }
    centre_label_short = {
                    "Bary" : "Centroid", 
                    "Circ" : "CC"
        }

    # matplotlib.rcParams['text.usetex'] = True


def load_samples():
    my_sample={}

    for file in Gals_list+Groups_list:
        my_sample[file] = pd.read_csv(datafolder+file+".csv")
        print("%i objects loaded from %s"%(my_sample[file].shape[0],file))
    return my_sample


def load_PC():
    my_PC={}

    for file in ['Gals','Groups']:
        my_PC[file] = pd.read_csv(datafolder+"PC_"+file+".csv")
        print("%i objects loaded from %s"%(my_PC[file].shape[0],file))
    return my_PC

    
def Mag_abs(mag,z) :
    #d_lum in Mpc, so 1E5 in units of 10pc
    return mag - 5*np.log10(cosmo.luminosity_distance(z)/(10*u.parsec)) #d_lum in Mpc


def Lum(df,Mag='M_r'):
    return(10**(-0.4*(df[Mag]- Mr_Sol))) 

def complete_agg(x):    
    # function for intermediate selections (group is complete if all members are)
    complete = (x.loc[~x['complete']].shape[0]==0)
    return pd.Series([complete], 
                     index=['complete'])

def coordBGG(x): 
    x = x.sort_values('M_r')
    RA_BGG = x.head(1)['RA'].values[0]
    Dec_BGG = x.head(1)['Dec'].values[0]
    M_BGG = x.head(1)['M_r'].values[0]
    return pd.Series([RA_BGG, Dec_BGG,M_BGG], 
                 index=['RA_BGG', 'Dec_BGG','M_BGG']) 

def rank(df,quantity,Group="Group"):
    return df.groupby(Group)[quantity].rank(ascending=1).astype(int)



def dist2BGG(df,rank,Group='Group',RA='RA',Dec='Dec',RA_BGG='RA_BGG',Dec_BGG='Dec_BGG'):

    return df.apply(lambda x:0.0 if (x[rank]==1) else 
                    su.calc_sep(su.d2r(x[RA]),su.d2r(x[Dec]),
                                su.d2r(x[RA_BGG]),su.d2r(x[Dec_BGG]))*3600,
                    axis=1)


def sSFRline(lg_M):
# Defining sSFR limit from Knobel+15
    return  (-0.3*lg_M - 7.85) # Msol/Gyr


def SFRexcess(sSFR,lgm_tot_p50):
    return (sSFR - sSFRline(lgm_tot_p50))

def SFRcategory(SFRexcess):
    conditions = [
        (SFRexcess>sSFRwidth),
        (SFRexcess<sSFRwidth) & (SFRexcess>-sSFRwidth),
        (SFRexcess<-sSFRwidth)]
    choices = ['M', 'G', 'Q']
    return np.select(conditions, choices, default='X')

def bin_SFRcategory(SFRexcess):
    conditions = [
        (SFRexcess>0),
        (SFRexcess<0)]
    choices = ['M', 'Q']
    return np.select(conditions, choices, default='X')



def mass(galaxies,groupnum):
    lgmass = galaxies.loc[galaxies['Group']==groupnum]['lgm_tot_p50']
    mass = 10**lgmass
    return mass.sum()

# Calc r_200 in kpc

def M_tilde(x):
    # print(f"   In M_tilde: x={x}")
    return (np.log(x+1)-x/(x+1))/(np.log(2)-1/2)

def r_200_kpc(M_200): # in kpc
    return 432*(M_200/(1e13))**(1/3)
    # return (G*M_200/(100*H0**2))**(1/3) 


# c_200 from Dutton & Maccio 14:

def c_LCDM(M,z):
    norm = 10**( 0.520+(0.905-0.520)*np.exp(-0.617*z**1.21) )
    slope = -0.101+0.026*z  
    return norm*(h*M/(10**12))**slope  # voir si 10^12*M_sun nécessaire (M_Sun à créer)

def E(z):
    return cosmo.H(z).value/H0

def EqA11(M_200,*args):
    lM180m,z = args
    M_Y = 10**lM180m
    f1 = (0.9*Om0)**(-1/3)
    f2 = c_LCDM(M_200,z)
    f3 = E(z)**(2/3)/(1+z)
    f4 = (M_Y/M_200)**(1/3)
    f5 = M_tilde(f2)
    return np.log10(M_tilde(f1*f2*f3*f4/f5)*M_200) - np.log10(M_Y)


################################################
def Velocity(z,zg):
    return c*(z-zg)/(1+zg)




def Group_agg(y, z_group_label=None, Id="Id", z='z', seed=1e12, circ=False, morph=False, sSFR=False):
#Defining Aggregation for groups building
          
    x = y.sort_values('M_r').reset_index()
    # if 'Lum' not in x.columns:
    #     x['Lum'] = pu.Lum(x)
    x['my_Lum'] = pu.Lum(x['M_r'])
    Lum_BGG = x.iloc[0]['my_Lum']
    Lum_group = sum(x['my_Lum'])
    M_group = pu.Mag(Lum_group)
    FracLumBGG = Lum_BGG/Lum_group
    RA_BGG = x.iloc[0]['RA']
    Dec_BGG = x.iloc[0]['Dec']
    z_mean = x[z].mean()
    x['V'] = Velocity(x[z],z_mean)
    V_BGG = x.iloc[0]['V']
    V_moy = x['V'].mean()
    if morph:
        Morph_BGG = x.iloc[0]['MorphZoo']

        
#    V_err = np.std(x['err_RV']) #RMS
    diff = x.iloc[1]['M_r'] - x.iloc[0]['M_r']

    if sSFR:
        all_SFR = ~(x['SFRcategory']=='X').any()
        BGG_SFRcategory = x.iloc[0]['SFRcategory']
        Prop_M_Sat = (x['SFRcategory'].iloc[1:] == 'M').mean()
        Prop_M_Tot = (x['SFRcategory'].iloc[:] == 'M').mean()
        Prop_G_Sat = (x['SFRcategory'].iloc[1:] == 'G').mean()
        Prop_G_Tot = (x['SFRcategory'].iloc[:] == 'G').mean()
        Prop_Q_Sat = (x['SFRcategory'].iloc[1:] == 'Q').mean()
        Prop_Q_Tot = (x['SFRcategory'].iloc[:] == 'Q').mean()

    nbgal = x.shape[0]      


    (RA_bary, Dec_bary) = su.calc_bary(x)
    Radius_Bary_arcmin = su.calc_diameter_arcmin(x)
    Offset_bary_abs_arcmin = su.r2d(su.calc_sep(su.d2r(RA_bary),su.d2r(Dec_bary),su.d2r(RA_BGG),su.d2r(Dec_BGG)))*60
    Offset_bary = Offset_bary_abs_arcmin / Radius_Bary_arcmin

    if circ:
        (RA_Circ, Dec_Circ,Radius_Circ_arcmin) = su.hcirc_sph(x)
        Offset_Circ_abs_arcmin = su.r2d(su.calc_sep(su.d2r(RA_Circ),su.d2r(Dec_Circ),su.d2r(RA_BGG),su.d2r(Dec_BGG)))*60
        Offset_Circ = np.clip(0,1,Offset_Circ_abs_arcmin / Radius_Circ_arcmin)


    
    Vdisp = pu.V_disp_gapper(x)

#     # Possibility to discard groups 
#     # containing galaxies without error on velocity or with an error  > sigma_gapper/sqrt(2)
#     if (x['err_RV'].max()>sigma/np.sqrt(2) or x['err_RV'].min()<0):
#         Bad_err_RV = 'Bad'
#     else:
#         Bad_err_RV = 'Good'

    Voffset = np.abs(V_BGG-V_moy)/Vdisp

    
    
    
    if z_group_label is not None:
        z_group = x.iloc[0][z_group_label]
    else:
        z_group = z_mean 
    
    arcmin_to_rad = np.pi/(180*60)
    Dist_Group_Mpc = cosmo.luminosity_distance(z_group)
    size_Group_Bary_kpc =  (Radius_Bary_arcmin * arcmin_to_rad * Dist_Group_Mpc).to(u.kpc).value
    if circ:
        size_Group_Circ_kpc =  (Radius_Circ_arcmin * arcmin_to_rad * Dist_Group_Mpc).to(u.kpc).value


    M_virial = pu.M_virial(x,cosmo,Vdisp)
    t_cr = pu.t_cr(size_Group_Bary_kpc,Vdisp)

    values = [Lum_BGG, Lum_group, FracLumBGG, z_group, diff,  
                    nbgal, RA_BGG, Dec_BGG,RA_bary, Dec_bary,Radius_Bary_arcmin,
                    Offset_bary, V_BGG, V_moy, Vdisp,Voffset,size_Group_Bary_kpc, M_group, M_virial, M_virial/Lum_group,t_cr] 
    labels = ['Lum_BGG', 'Lum_group', 'FracLumBGG', 'z_group', 
                    'DeltaR12', 'NbGal', 
                    'RA_BGG', 'Dec_BGG', 'RA_Bary', 'Dec_Bary', 'Radius_Bary_arcmin',
                    'Offset_Bary', 'V_BGG', 'V_moy',
                    'Vdisp','Voffset', 'size_Group_Bary_kpc', 'M_group', 'M_virial', 'M_virial_over_L', 't_cr']
    
    if circ:
        values += [RA_Circ, Dec_Circ,Radius_Circ_arcmin, Offset_Circ,size_Group_Circ_kpc]
        labels += ['RA_Circ', 'Dec_Circ','Radius_Circ_arcmin', 'Offset_Circ','size_Group_Circ_kpc']
    
    if morph:
        values += [Morph_BGG]
        labels += ['Morph_BGG']
        
    if sSFR:
        values += [BGG_SFRcategory, 
                    all_SFR,Prop_M_Sat, Prop_M_Tot, Prop_G_Sat, Prop_G_Tot, 
                    Prop_Q_Sat, Prop_Q_Tot]
        labels += ['BGG_SFRcategory', 
                    'all_SFR','Prop_M_Sat', 'Prop_M_Tot', 'Prop_G_Sat', 'Prop_G_Tot' , 
                    'Prop_Q_Sat', 'Prop_Q_Tot']
        

        
    return pd.Series(values, index=labels) 





def BuidSelector(col,labelinf,labelsup):
    medGapMag = col.median()
    return col.apply(lambda x: labelsup if x > medGapMag else labelinf)

def good_sfr(ssfr,lgm_tot_p50):
    for col in [ssfr,lgm_tot_p50]:
        col = np.where(col.isnull(),-9999,col)
    return ((ssfr>=-25) & (ssfr<=-5) & 
            (lgm_tot_p50>=5)  & (lgm_tot_p50<=14))

def match_cat(df1,df2,ra1="RA",dec1="Dec",ra2="RA",dec2="Dec",suff="2"):
    Astrodf1 = Table.from_pandas(df1[[ra1,dec1]])
    Astrodf2 = Table.from_pandas(df2[[ra2,dec2]])

    c = SkyCoord(Astrodf1[ra1]*u.deg, Astrodf1[dec1]*u.deg)
    catalog = SkyCoord(Astrodf2[ra2]*u.deg, Astrodf2[dec2]*u.deg)
    idx, d2d, d3d = c.match_to_catalog_3d(catalog)
    catalog_matches = df2.iloc[idx].reset_index(drop=True).copy()
    catalog_matches['Separation']=d2d.arcsec
    matched = pd.concat([df1.reset_index(drop=True), catalog_matches.add_suffix("_"+suff)], axis=1).copy()  
    
    return matched



def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2} h {:0>2} min {:05.2f} s".format(int(hours),int(minutes),seconds))


def MorphZoo(df):
    Morph_conditions = [
        (df['spiral'] == 1),
        (df['elliptical'] == 1),
        (df['uncertain'] == 1),
        (df['spiral'].isnull())
        ]

    Morph_values = ['S', 'E', 'U', 'X']

    return pd.Series(np.select(Morph_conditions, Morph_values))