def formatted_label(i,lumG=False):
    switcher={
        'LumFrac'             : r'$L_{\mathrm{BGG}}/L_{\mathrm{Group}}$',
        'FracLumBGG'          : r'$L_{\mathrm{BGG}}/L_{\mathrm{Group}}$',
        'frac_lum'            : r'$\mathrm{L}/\mathrm{L}_\mathrm{group}$',
        'frac_mass'           : r'$\mathcal{M}/\mathcal{M}_\mathrm{group}$',
        'frac_mass'           : r'$\mathcal{M}/\mathcal{M}_\mathrm{group}$',
# =============================================================================
        'Offset_Bary'         : r"$\Delta_\mathrm{BGG-cen}/ \\langle R_{ij} \\rangle$",
        'frac_radius_Bary'    : r"$\Delta_\mathrm{BGG-cen}/ \\langle R_{ij} \\rangle$",
        'Offset_Bary_gal'     : r"$\Delta_\mathrm{gal-cen}/ \\langle R_{ij} \\rangle$",
        'frac_radius_Bary_gal': r"$\Delta_\mathrm{gal-cen}/ \\langle R_{ij} \\rangle$",
        'Offset_Circ'         : r'$\Delta_\mathrm{Circ}/R_\mathrm{Circ}$',
        'frac_radius_Circ'    : r'$\Delta_\mathrm{Circ}/R_\mathrm{Circ}$',
        'Vdisp'               : r'$\sigma_v$ (km s$^{-1}$)',
        'Voffset'             : r'Velocity offset',
# =============================================================================
        # 'Voffset'             : '$\kappa$',
        'Voffset'             : r'$\Delta V_\mathrm{BGG}/\sigma_v$',
        'DeltaR12'            : r'$\Delta M_{r12}$',
        'PropS'               : r' Proportion of type S satellites',
        'Radius_Bary_kpc'     : r"$\\langle R_{ij} \\rangle$ (kpc)",
        'Radius_Circ_kpc'     : r"Circular radius (kpc)",
        'BGG_SFRcategory'     : r'BGG SFR category',
        'BGG_Morph'           : r'BGG Morphology',
        "Non dominated"       : r"Non dominated",
        "Dominated"           : r"Dominated", 
        "BCentered"           : r"Centered Centroid",
        "BMisfit"             : r"Off-center Centroid", 
        "CCentered"           : r"Centered Circle",
        "CMisfit"             : r"Off-center Circle",
        "VCentered"           : r"Medium BGG velocity",
        "VMisfit"             : r"Extreme BGG velocity", 
        "Misfit_Bary"         : r"Centroid position",
        "Misfit_Circ"         : r"Circle position",
        "Vmisfit"             : r"BGG radial velocity",
        "Q"                   : r"Quenched",
        "M"                   : r"Main sequence",
        "G"                   : r"Green valley",
        'size_Group_Bary_kpc' : r"$\\left\\langle R_{ij}\\right\\rangle$ (kpc)",
        'size_Group_Circ_kpc' : r"$R_\mathrm{Circ}$ (kpc)",
        'Lum'                 : r"Luminosity ($L_\odot$)",
#         'GSL'                 : "Luminosity ($10^9 \times L_{\odot}$)",
        'GSL'                 : r"Luminosity ($G L_{\odot}$)",
#         'GroupGSL'            : "Group Luminosity ($10^9 \times L_{\odot}$)"
        'GroupGSL'            : r"Group Luminosity ($G L_{\odot}$)",
        'M_r'                 : r"$M_r$",
        'Mr_BGG'              : r"$Mr_\mathrm{BGG}$",
        'M_group'             : r"$M_{r,\mathrm{Group}}$",
        'lMass_group'         : r"$\log(\mathcal{M}_\mathrm{Group}/\mathcal{M}_\odot)$",
        'z_group'             : r"$ \overline{z}_\mathrm{Group}$",
        'r_200'               : r"$R_{200}$ (kpc)",
        'Mass_200'            : r"$\mathcal{M}_{200}/\mathcal{M}_\odot$",
        'lMass_200'           : r"$\log(\mathcal{M}_{200}/\mathcal{M}_\odot)$",
        'Morph_BGG'           : r"BGG Morphology from Zoo 1",
        'S'                   : r'Spiral',
        'E'                   : r'Elliptical',
        'S0'                  : r'Lenticular',
        't_cr'                : r'$t_\mathrm{cr}$ (Gyr)',
        'M_over_Lr'           : r'$M/L_r$ (solar)',
        'M_virial'            : r'$\mathcal{M}_\mathrm{VT}/(10^{11} M_\odot)$',
        'logM_virial'         : r'$\log(\mathcal{M}_\mathrm{VT}/M_\odot$)',
        'M_virial_over_L'     : r'$\mathcal{M}_\mathrm{VT}/L_r$ (solar)',
        'logLum_group'        : r'$\log(L_{r,\mathrm{group}}/\mathrm{L_\odot})$',
        'Prop_M_Sat'          : r'$Frac_\mathrm{Sat}(\mathrm{BGG} = M)$',
        'Prop_Q_Sat'          : r'$Frac_\mathrm{Sat}(\mathrm{BGG} = Q)$',
        'specsfr_tot_p50'     : r'sSFR (yr$^{-1}$)',
        'sSFR'                : r'sSFR (yr$^{-1}$)',
        'lgm_tot_p50'         : r'$\log(M_\star/M_\odot)$',
        'lgm'                 : r'$\log(M_★/M_\odot)$'
     }
    
    lumswitch={
        'Lum_BGG'            : r'BGG luminosity ($L_\odot$)',
        'log_Lum_BGG'        : r'log$_{10}$(BGG luminosity/$L_\odot$)',
        # 'Lum_group'          : 'Group luminosity ($L_\odot$)',
        'Lum_group'          : r' $L_\mathrm{group}$ ($L_\odot$)',
        # 'log_Lum_group'      : 'log$_{10}$(Group luminosity/$L_\odot$)',
        'log_Lum_group'      : r'$\log(L_\mathrm{group}/L_\odot)$',
        'Lum_Sat'            : r'Satellites luminosity ($L_\odot$)',
        'log_Lum_Sat'        : r'log$_{10}$(Satellites luminosity/$L_\odot$)',
        }
    
    lumGswitch = {
        'Lum_BGG'            : r'BGG luminosity ($10^9 L_\odot$)',
        'log_Lum_BGG'        : r'log$_{10}$(BGG luminosity/$10^9 L_\odot$)',
        'Lum_group'          : r'Group luminosity ($10^9 L_\odot$)',
        'log_Lum_group'      : r'log$_{10}$(Group luminosity/$10^9 L_\odot$)',
        'Lum_Sat'            : r'Satellites luminosity ($10^9 L_\odot$)',
        'log_Lum_Sat'        : r'log$_{10}$(Satellites luminosity/$10^9 L_\odot$)',        
        }
    
    if lumG:
        switcher.update(lumGswitch)
    else:
        switcher.update(lumswitch)

    
    return switcher.get(i,"Invalid label")



def my_label(i):
    switcher={
        "CG"                 : "CG$_4$",
        "YGB"                : "Yang$_{4B}$",
        "YGC"                : "Yang$_{4C}$",
        "Dominated"          : "Dominated groups",
        "Non dominated"      : "Not dominated groups",
    }
    return switcher.get(i,"Invalid label")

def short_label(i):
    switcher={
        "Dominated"          : "Dom.",
        "Non dominated"      : "Not dom.",
        "VMisfit"            : "V misfit",
        "VCentered"          : "V centered",
        "BMisfit"            : "Bary. misfit",
        "BCentered"          : "Bary. centered",
        "CMisfit"            : "Circ. misfit",
        "CCentered"          : "Circ. centered"

    }
    return switcher.get(i,"Invalid label")

def morph_marker(morph):
    """
    Returns the marker for a given morphology type.
    """
    switcher = {
        'Spiral': 'o',
        'Elliptical': 's',
        'Uncertain': 'x',
        'Lenticular': '^',
        'Irregular': 'D',
    }
    return switcher.get(morph, 'o')  # Default to 'o' if morphology is not recognized

def morph_color(morph):
    """
    Returns the color for a given morphology type.
    """
    switcher = {
        'Spiral': 'blue',
        'Elliptical': 'red',
        'Uncertain': 'gray',
        'Lenticular': 'green',
        'Irregular': 'purple',
    }
    return switcher.get(morph, 'blue')  # Default to 'blue' if morphology is not recognized
