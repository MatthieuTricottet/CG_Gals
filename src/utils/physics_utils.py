from . import spherical_utils as su

def virial_theorem_mass(gals,cosmo,Vdisp,debug=False): # Vdisp in km/s, returns virial mass in units of 10^11 Msol
    """Computes the mass of a group of galaxies estimated from the virial theorem

    Args:
        gals (pandas.DataFrame): galaxies in the group, must have columns 'RA', 'Dec' and 'z'
        cosmo (astropy.cosmology): [description]
        Vdisp (real): velocity dispersion of the group in km/s
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        real: mass in units of 10^11 Msol
    """

    perm=pd.DataFrame(list(it.combinations(np.arange(gals.shape[0]), 2)))
    inverse_separation = pd.Series(dtype='float64')
    for index, row in perm.iterrows():
        phi1 = su.d2r(gals.iloc[row[0]]['RA']) 
        theta1 = su.d2r(gals.iloc[row[0]]['Dec']) 
        phi2 = su.d2r(gals.iloc[row[1]]['RA']) 
        theta2 = su.d2r(gals.iloc[row[1]]['Dec']) 
        separation = su.calc_sep(phi1,theta1,phi2,theta2)
        if debug:
            print(f'separation {index}: {su.r2arcmin(separation):.3f} arcmin')

        inverse_separation = mu.Series_append(inverse_separation,1/separation)
    
    mean_inv_sep = inverse_separation.mean() 
    R_harm_angl = 1/mean_inv_sep
    if debug:
        print(f'R_harm_angl = {R_harm_angl:.3f} rd = {su.r2arcmin(R_harm_angl):.3f} arcmin')

    z_group = gals['z'].mean()
    # Dist_Group_kpc = cosmo.luminosity_distance(z_group)
    Dist_Group_kpc = cosmo.angular_diameter_distance(z_group).to(u.kpc)
    R_harm_kpc = (R_harm_angl * Dist_Group_kpc).value

    if debug:
        print(f'R_harm_kpc = {R_harm_kpc:.3f}')

    
    M_virial = 1e11 * 3 * np.pi * R_harm_kpc * (Vdisp/v0)**2 / G   

    return M_virial

def crossing_time(Rij_kpc,sigma_v_kms):     # crossing time; sizes = <Rij> in kpc, sigma_v in km/s
    """Computes the crossing time of a system

    Args:
        Rij_kpc (real, list, np.array, pandas series...): caracteristic size of the system in kpc
        sigma_v_kms (matching Rij_kpc): velocity dispersion of the system in km/s

    Returns:
        matching Rij_kpc type: crossing time in Gyr
    """
    return 0.887*Rij_kpc/sigma_v_kms
    