#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:41:05 2020

@author: matt
"""
import pandas as pd
import numpy as np
import itertools as it
from . import pandas_utils as pu 

from astropy.coordinates import angular_separation


#deg to rad
def d2r(x):
    return x * np.pi / 180.0

#arcmin to rad
def a2r(x):
    return x * np.pi / 10800.0

#rad to deg
def r2d(x):
    return x * 180.0 / np.pi

#rad to arcmin
def r2arcmin(x):
    return r2d(x)*60

#arcmin to rad
def arcmin2r(x):
    return d2r(x/60)

# sin in deg
def sind(x):
    return(np.sin(d2r(x)))

# cos in deg
def cosd(x):
    return(np.cos(d2r(x)))

# tan in deg
def tand(x):
    return(np.tan(d2r(x)))

# sin in arcmin
def sina(x):
    return(np.sin(a2r(x)))

# cos in arcmin
def cosa(x):
    return(np.cos(a2r(x)))

# tan in arcmin
def tana(x):
    return(np.tan(a2r(x)))

# asin in arcmin
def asina(x):
    return(r2arcmin(np.arcsin(x)))

# acos in arcmin
def acosa(x):
    return(r2arcmin(np.arccos(x)))

# asin in deg
def asind(x):
    return(r2d(np.arcsin(x)))

# acos in deg
def acosd(x):
    return(r2d(np.arccos(x)))

# atan in deg
def atand(x):
    return(r2d(np.arctan(x)))


# Computes separation on a sphere in radian 
# NB: usage seems to be RA = phi and Dec = theta
def calc_sep(phi1,theta1,phi2,theta2):
    return np.arccos(np.sin(theta1)*np.sin(theta2) + 
                   np.cos(theta1)*np.cos(theta2)*np.cos(phi2 - phi1) )

# angle in rad between two (alpha,delta) pairs in rad
# roughly the same than previous, but adapted for circumscribed circles


# Like calc_sep, but result in deg
def calc_sep_deg(phi1,theta1,phi2,theta2):
        # phi1r = d2r(phi1) 
        # theta1r = d2r(theta1) 
        # phi2r = d2r(phi2) 
        # theta2r = d2r(theta2) 
        phi1r = np.radians(phi1) 
        theta1r = np.radians(theta1) 
        phi2r = np.radians(phi2) 
        theta2r = np.radians(theta2) 
        return(np.degrees(calc_sep(phi1r,theta1r,phi2r,theta2r)))


# Like calc_sep, but result in arcmin
def calc_sep_arcmin(phi1,theta1,phi2,theta2):
        # phi1a = d2r(phi1) 
        # theta1a = d2r(theta1) 
        # phi2a = d2r(phi2) 
        # theta2a = d2r(theta2) 
        # return(r2arcmin(calc_sep(phi1a,theta1a,phi2a,theta2a)))
    return 60.0*angular_separation(phi1,theta1,phi2,theta2)

def angles(df):
    cosangle = np.cos(df['Dec1'])*np.cos(df['Dec2'])*np.cos(df['RA1']-df['RA2']) + \
               np.sin(df['Dec1'])*np.sin(df['Dec2'])
    cosangle=limit_trig(cosangle)
    return np.arccos(cosangle)

def angles_to_centre(df,RA_cen,Dec_cen):
    cosangle = np.cos(df['Dec'])*np.cos(Dec_cen)*np.cos(df['RA']-RA_cen) + \
               np.sin(df['Dec'])*np.sin(Dec_cen)
    cosangle=limit_trig(cosangle)
    return np.arccos(cosangle)

#Defines barycenter of a group
def calc_bary(group):
    phi = d2r(group['RA'])
    theta = d2r(group['Dec'])
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    
    x_O, y_O, z_O = x.mean(), y.mean(), z.mean() # To be modified
    # when mass problem (related to SDSS identification) is settled
    #(x * group['Mass']).sum() / group['Mass']).sum()
    
    l = np.sqrt(x_O**2 + y_O**2 + z_O**2)
    
    (x_M, y_M, z_M) = (x_O, y_O, z_O)/l
    
    Dec_M = np.arcsin(z_M)
    if (y_M <0):
        RA_M = 2.0*np.pi - np.arccos(x_M / np.cos(Dec_M))
    else:
        RA_M = np.arccos(x_M / np.cos(Dec_M))

    return (r2d(RA_M), r2d(Dec_M))



#Computes projected radius of group based on barycenter
# def calc_diameter_arcmin(group,RA_bary, Dec_bary,Id='Id'):
def calc_diameter_arcmin(group):
    perm=pd.DataFrame(list(it.combinations(np.arange(group.shape[0]), 2)))
    separation = pd.Series(dtype='float64')
    for index, row in perm.iterrows():
        phi1 = d2r(group.iloc[row[0]]['RA']) 
        theta1 = d2r(group.iloc[row[0]]['Dec']) 
        phi2 = d2r(group.iloc[row[1]]['RA']) 
        theta2 = d2r(group.iloc[row[1]]['Dec']) 

        sep = r2d(calc_sep(phi1,theta1,phi2,theta2)) 
        separation = pu.Series_append(separation,sep)
    return separation.median() * 60.0



# input: Pandas.Series of sin and cos ; output: value trimed to [-1,1]
def limit_trig(df):
    df.loc[df>1.0] = 1.0
    df.loc[df<-1.0] = -1.0
    return df



# midpoint in spherical trigonometry
# Radians
def midpoint_sph(alpha1, delta1, alpha2, delta2):
    _Bx = np.cos(delta2)*np.cos(alpha2-alpha1)
    _By = np.cos(delta2)*np.sin(alpha2-alpha1)
    deltaM = np.arctan2(np.sin(delta1)+np.sin(delta2),np.sqrt((np.cos(delta1)+_Bx)**2 + _By**2))
    alphaM = alpha1 + np.arctan2(_By,_Bx+np.cos(delta1))
    
    return(alphaM,deltaM)



# midpoint in spherical trigonometry
# Degres
def midpoint_sph_d(alpha1, delta1, alpha2, delta2):
    alpha1r = d2r(alpha1)
    delta1r = d2r(delta1)
    alpha2r = d2r(alpha2)
    delta2r = d2r(delta2)
    
    _Bx = np.cos(delta2r)*np.cos(alpha2r-alpha1r)
    _By = np.cos(delta2r)*np.sin(alpha2r-alpha1r)
    deltaM = np.arctan2(np.sin(delta1r)+np.sin(delta2r),np.sqrt((np.cos(delta1r)+_Bx)**2 + _By**2))
    alphaM = alpha1r + np.arctan2(_By,_Bx+np.cos(delta1r))
     
    return(r2d(alphaM),r2d(deltaM))




#Cartesian product
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))

def cartesian_product_basic(left, right):
    # Usage: df_cartesian = cartesian_product_basic(*[df1, df2])
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))


# Smallest circumscribed circle (spherical)
# def hcirc_sph(group,Id="Id"):
def hcirc_sph(group,Id="Id"):

    # print("Entering hcirc_sph")
    # print(group[['RA','Dec']])

    eps = 1.e-6
        
    # find maximum separation
    separation = pd.DataFrame(columns=["Point1","Point2","sep"])
    for pair in it.combinations(np.arange(group.shape[0]), 2):
        RA1   = group.iloc[pair[0]]['RA']
        Dec1  = group.iloc[pair[0]]['Dec']
        RA2   = group.iloc[pair[1]]['RA']
        Dec2  = group.iloc[pair[1]]['Dec']

        separation.loc[separation.shape[0]]=[pair[0],pair[1],calc_sep_arcmin(RA1,Dec1,RA2,Dec2)]


    separation.sort_values(by='sep',ascending=False, inplace=True)
        
    i0 = int(separation.iloc[0]['Point1'])
    i1 = int(separation.iloc[0]['Point2'])
    RA1   = group.iloc[i0]['RA']
    Dec1  = group.iloc[i0]['Dec']
    RA2   = group.iloc[i1]['RA']
    Dec2  = group.iloc[i1]['Dec']
    (RAm,Decm) = midpoint_sph_d(RA1,Dec1,RA2,Dec2)
    theta = separation.iloc[0]['sep']/2
    
    DistCen = pd.DataFrame(columns=["Point","sep"])
    for i in np.arange(group.shape[0]):
        if ((i!=i0) and (i!=i1)):
            (RA,Dec) = (group.iloc[i]['RA'],group.iloc[i]['Dec'])
            dist = calc_sep_arcmin(RAm,Decm,RA,Dec)
            DistCen.loc[DistCen.shape[0]]=[i,dist]


    if (DistCen.loc[DistCen['sep']>theta].shape[0] ==0):
        # print("Exiting hcirc_sph - from DistCen[sep]>theta\n\n")
        return pd.Series({ 'RA_cen_Circ'        : RAm,
                           'Dec_cen_Circ'       : Decm,
                           'Radius_Circ_arcmin' : theta})
    else:
        # print('   In else')
        # print(group[Id].to_list())
        for triangle in it.combinations(group[Id].to_list(), 3):
            # print(f'      triangle: {triangle}')
            gr3 = group.loc[group[Id].isin(triangle)]            
            others = group.loc[~group[Id].isin(triangle)][[Id, 'RA', 'Dec']].copy()

            # print(f'      Entering circ3_sph')
            # print(gr3)
            circle = circ3_sph(gr3,Id)
            # print(f'      Out of circ3_sph')

            if(others.shape[0]==0):
                # print("Exiting hcirc_sph - from shape==0\n\n")
                return circle
            else:
                RAm = circle['RA_cen_Circ'] 
                Decm = circle['Dec_cen_Circ'] 
                others['dist2Cen_arcmin'] = calc_sep_arcmin(RAm,Decm,others['RA'],others['Dec'])
                if (others['dist2Cen_arcmin'].max()<= circle['Radius_Circ_arcmin']+eps):
                    # print("Exiting hcirc_sph - from max<=circle\n\n")
                    return circle
                
                
    print("Error in computing circumscribed circle of group:")
    print(group[['RA','Dec']])
    return pd.Series({'RA_cen_Circ'        : -999999,
                      'Dec_cen_Circ'       : -999999,
                      'Radius_Circ_arcmin' : -999999})





# Smallest circumscribed circle for 3 points (spherical) - called by hcirc_sph
def circ3_sph(group,Id="Id"): 
    # print(f'         In circ3_sph')

    separation = pd.DataFrame(columns=['Point1','Point2','RA1','Dec1','RA2','Dec2','sep'])
    

    for pair in it.combinations(np.arange(group.shape[0]), 2):
        RA1   = d2r(group.iloc[pair[0]]['RA'])
        Dec1  = d2r(group.iloc[pair[0]]['Dec'])
        RA2   = d2r(group.iloc[pair[1]]['RA'])
        Dec2  = d2r(group.iloc[pair[1]]['Dec'])

        separation.loc[separation.shape[0]]=[pair[0],pair[1],RA1,Dec1,RA2,Dec2,
                                             calc_sep(RA1,Dec1,RA2,Dec2)]
        
    separation.sort_values(by='sep',inplace=True,ignore_index=True)
    halfsepmax = separation.iloc[2]['sep']/2
    
    ############################
    
    if (2*np.cos(halfsepmax)**2 < np.cos(separation.iloc[0]['sep']) + np.cos(separation.iloc[1]['sep'])):
        RA0,Dec0 = midpoint_sph(separation.iloc[2]['RA1'],separation.iloc[2]['Dec1'],separation.iloc[2]['RA2'],separation.iloc[2]['Dec2'])
        # print(f'         Exiting circ3_sph from cosine condition')
        return pd.Series({'RA_cen_Circ'        : r2d(RA0),
                          'Dec_cen_Circ'       : r2d(Dec0),
                          'Radius_Circ_arcmin' : r2arcmin(halfsepmax)})

    
    ############################
    
    s = 0.5 * separation['sep'].sum()
        
    tanr = np.sqrt(np.sin(s-separation.iloc[0]['sep']) *
                   np.sin(s-separation.iloc[1]['sep']) *
                   np.sin(s-separation.iloc[2]['sep']) / np.sin(s))
    ang = 2*np.arctan(tanr/np.sin(s-separation['sep']))

    S = 0.5 * ang.sum()
    
    theta = np.arctan(np.tan(separation.iloc[0]['sep']/2)/np.cos(S-ang.iloc[0]))
    
    
    RA1 = separation.iloc[0]['RA1']
    Dec1 = separation.iloc[0]['Dec1']
    RA2 = separation.iloc[0]['RA2']
    Dec2 = separation.iloc[0]['Dec2']
    sep0 = separation.iloc[0]['sep']
       
    
    cosNBC = (np.sin(Dec1)-np.sin(Dec2)*np.cos(sep0))/(np.cos(Dec2)*np.sin(sep0))
    cosNBC = np.clip(-1,1,cosNBC)
    cosOBC = np.cos(theta)*(1-np.cos(sep0))/(np.sin(theta)*np.sin(sep0))
    cosOBC = np.clip(-1,1,cosOBC)
    
    
    sign = pd.DataFrame([-1,1])
    deltadelta = np.arccos(cosOBC)
    sindelta0 = np.sin(Dec2)*np.cos(theta) + \
                np.cos(Dec2)*np.sin(theta)*np.cos(np.arccos(cosNBC)-sign*deltadelta)
    sindelta0 = np.clip(-1,1,sindelta0)

    delta0t =  np.arcsin(sindelta0)

    cosdelalpha  = (np.cos(theta)-np.sin(Dec2)*sindelta0)/(np.cos(Dec2)*np.cos(delta0t))
    cosdelalpha.clip(-1, 1, inplace=True)

    deltaalpha = np.arccos(cosdelalpha)
    alpha0p = RA2 + deltaalpha
    alpha0m = RA2 - deltaalpha

    # Choose one of four values by checking that points fit on circle
    
    PossiblePoints = pd.DataFrame(columns={'RA0','Dec0'})
    PossiblePoints['RA0'] = pd.concat([alpha0p, alpha0m], ignore_index=True).squeeze()
    PossiblePoints['Dec0'] = pd.concat([delta0t, delta0t], ignore_index=True).squeeze()

    my_group = group.copy()
    my_group['RA_rad'] = d2r(my_group['RA'])
    my_group['Dec_rad'] = d2r(my_group['Dec'])
    listdistmax = pd.DataFrame(columns=['Index','adiffsepmax'])
    for indexCenter,rowCenter in PossiblePoints.iterrows():
        distmax = 0
        for indexPoints,rowPoints in my_group.iterrows():
            dist = calc_sep(rowCenter['RA0'],    rowCenter['Dec0'],
                            rowPoints['RA_rad'], rowPoints['Dec_rad'])
            if (dist>distmax):
                distmax = dist
        listdistmax = pu.df_append(listdistmax,{'Index'      : indexCenter, 
                                          'adiffsepmax': np.abs(theta-distmax)})
        
    # if (listdistmax.shape[0]==0):
    #     print("Problem 1!")
    # print("YOU ARE WHERE YOU THINK")
    # print(listdistmax)
    
    # metaindex = listdistmax['adiffsepmax'].idxmin()  # Seems bugged!!!!!
    
    metaindex = listdistmax['adiffsepmax'].sort_values().drop_duplicates(keep='last').index[0]
    
    # print("BEFORE METAINDEX")
    # print(f'metaindex: {metaindex}')
    bestindex = int(listdistmax.iloc[metaindex]['Index'])
    # print("AFTER bestindex")
    bestrow = PossiblePoints.iloc[bestindex]

    # print(f'         Exiting circ3_sph from the end')
    return pd.Series({'RA_cen_Circ'        : r2d(bestrow['RA0']),
                      'Dec_cen_Circ'       : r2d(bestrow['Dec0']),
                      'Radius_Circ_arcmin' : r2arcmin(theta)})



