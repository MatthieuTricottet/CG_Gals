import pandas as pd
import numpy as np
import itertools as it
import sys
from astropy import units as u

# sys.path.append('.')
# import spherical_utils as su # Matt

from . import spherical_utils as su

c = 299792.458
G = 43.012
Mr_Sol = 4.68
v0 = 100 # km/s


def Lum(Mag):
    """Luminosity from absolute magnitude

    Args:
        Mag (real, list, np.array, pandas series...): magnitude
    """
    return(10**(-0.4*(Mag- Mr_Sol)))


def Mag(Lum):
    """Absolute magnitude from luminosity

    Args:
        Lum (real, list, np.array, pandas series...): luminosity
    """
    return(-2.5*np.log10(Lum) + Mr_Sol)

def Velocity(z,zg):
    return c*(z-zg)/(1+zg)



def TremaineStats(Gap,MBGG):
    """Compute Tremaine - Richstone  T1 and T2 statistics
    From Tremaine, S. D. & Richstone, D. O. 1977, ApJ, 212, 311

    Args:
        Gap (pandas series): magnitude gaps between the BGG and the second brightest galaxy of the group
        MBGG (pandas series): magnitude of the BGG
    """
    MeanGap = Gap.mean()
    StdGap = Gap.std()
    StdMagBGG = MBGG.std()
    
    T1 = StdMagBGG/MeanGap
    T2 = StdGap/(MeanGap*np.sqrt(0.667))
    
    return[T1,T2]



def sSFRline(lgm):
# Defining sSFR limit from Knobel+15
    return  (-0.3*lgm - 7.85) # Msol/Gyr


def SFRexcess(sSFR,lgm):
    return (sSFR - sSFRline(lgm))

def SFRcategory(df,sSFR_label='specsfr_tot_p50',lgm_label='lgm_tot_p50', sSFRwidth = 0.4):
    sSFR = df[sSFR_label]
    lgm = df[lgm_label]
    excess = SFRexcess(sSFR,lgm)
    conditions = [
        (excess>sSFRwidth),
        (excess<sSFRwidth) & (excess>-sSFRwidth),
        (excess<-sSFRwidth)]
    choices = ['M', 'G', 'Q']
    return np.select(conditions, choices, default='X')

def bin_SFRcategory(SFRexcess):
    conditions = [
        (SFRexcess>0),
        (SFRexcess<0)]
    choices = ['M', 'Q']
    return np.select(conditions, choices, default='X')

 

def my_sSFRline(lgm):
    a = -0.56
    b = -5.55
    return  (a*lgm + b) # Msol/Gyr

def my_SFRexcess(sSFR,lgm):
    return (sSFR - my_sSFRline(lgm))


def my_SFRcategory(df,sSFR_label='specsfr_tot_p50',lgm_label='lgm_tot_p50', sSFRwidth = 0.4):
    sSFR = df[sSFR_label]
    lgm = df[lgm_label]
    excess = my_SFRexcess(sSFR,lgm)
    conditions = [
        (excess>0),
        (excess<=0)]
    choices = ['F', 'Q']
    return np.select(conditions, choices, default='X')


#* --------------------------------------------------------------------------------
#* Classify galaxies as AGN using a basic cut approximating Kewley et al. criteria.
#* --------------------------------------------------------------------------------
def classify_agn(row):
    """Returns True if the galaxy is classified as AGN based on NII/Ha & OIII/Hb."""
    if np.isnan(row['log_NII_Ha']) or np.isnan(row['log_OIII_Hb']):
        return False
    try:
        if row['log_NII_Ha'] > 0:
            return True
        elif (row['log_NII_Ha'] - 0.47) == 0:
            return False
        return row['log_OIII_Hb'] > (0.61 / (row['log_NII_Ha'] - 0.05) + 1.3)
    except Exception:
        return False