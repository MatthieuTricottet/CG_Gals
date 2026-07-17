def formatted_label(i,lumG=False):
    switcher={
        'LumFrac'             : r'$L_{\mathrm{BGG}}/L_{\mathrm{Group}}$',
        'FracLumBGG'          : r'$L_{\mathrm{BGG}}/L_{\mathrm{Group}}$',
        'frac_lum'            : r'$\mathrm{L}/\mathrm{L}_\mathrm{group}$',
        'frac_mass'           : r'$\mathcal{M}/\mathcal{M}_\mathrm{group}$',
        'frac_mass'           : r'$\mathcal{M}/\mathcal{M}_\mathrm{group}$',
# =============================================================================
        'Offset_Bary'         : r"$\Delta_\mathrm{BGG-cen}/ \langle R_{ij} \rangle$",
        'frac_radius_Bary'    : r"$\Delta_\mathrm{BGG-cen}/ \langle R_{ij} \rangle$",
        'Offset_Bary_gal'     : r"$\Delta_\mathrm{gal-cen}/ \langle R_{ij} \rangle$",
        'frac_radius_Bary_gal': r"$\Delta_\mathrm{gal-cen}/ \langle R_{ij} \rangle$",
        'Offset_Circ'         : r'$\Delta_\mathrm{Circ}/R_\mathrm{Circ}$',
        'frac_radius_Circ'    : r'$\Delta_\mathrm{Circ}/R_\mathrm{Circ}$',
        'Vdisp'               : r'$\sigma_v$ (km s$^{-1}$)',
        'Voffset'             : r'Velocity offset',
# =============================================================================
        # 'Voffset'             : '$\kappa$',
        'Voffset'             : r'$\Delta V_\mathrm{BGG}/\sigma_v$',
        'DeltaR12'            : r'$\Delta M_{r12}$',
        'PropS'               : r' Proportion of type S satellites',
        'Radius_Bary_kpc'     : r"$\langle R_{ij} \rangle$ (kpc)",
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
        'size_Group_Bary_kpc' : r"$\left\langle R_{ij}\right\rangle$ (kpc)",
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
        'logM_virial'         : r'$\log(\mathcal{M}_\mathrm{VT}/M_\odot)$',
        'M_virial_over_L'     : r'$\mathcal{M}_\mathrm{VT}/L_r$ (solar)',
        'logLum_group'        : r'$\log(L_{r,\mathrm{group}}/\mathrm{L_\odot})$',
        'Prop_M_Sat'          : r'$Frac_\mathrm{Sat}(\mathrm{BGG} = M)$',
        'Prop_Q_Sat'          : r'$Frac_\mathrm{Sat}(\mathrm{BGG} = Q)$',
        'specsfr_tot_p50'     : r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
        'sSFR'                : r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
        'lgm_tot_p50'         : r'$\log(M_\star/M_\odot)$',
        'lgm'                 : r'$\log(M_\star/M_\odot)$'
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

def formatted_text_label(i, lumG=False):
    """Return a prose label without units for narrative text."""

    switcher = {
        'Vdisp'               : r'$\sigma_v$',
        'Radius_Bary_kpc'     : r"$\langle R_{ij} \rangle$",
        'size_Group_Bary_kpc' : r"$\left\langle R_{ij}\right\rangle$",
        'Radius_Circ_kpc'     : r"$R_\mathrm{Circ}$",
        'size_Group_Circ_kpc' : r"$R_\mathrm{Circ}$",
        'r_200'               : r"$R_{200}$",
        't_cr'                : r'$t_\mathrm{cr}$',
        'M_over_Lr'           : r'$M/L_r$',
        'M_virial'            : r'$\mathcal{M}_\mathrm{VT}$',
        'M_virial_over_L'     : r'$\mathcal{M}_\mathrm{VT}/L_r$',
        'Lum'                 : r"Luminosity",
        'GSL'                 : r"Luminosity",
        'GroupGSL'            : r"Group luminosity",
        'Lum_BGG'             : r'BGG luminosity',
        'log_Lum_BGG'         : r'log$_{10}$(BGG luminosity)',
        'Lum_group'           : r'$L_\mathrm{group}$',
        'log_Lum_group'       : r'$\log(L_\mathrm{group})$',
        'Lum_Sat'             : r'Satellite luminosity',
        'log_Lum_Sat'         : r'log$_{10}$(satellite luminosity)',
        'specsfr_tot_p50'     : r'sSFR',
        'sSFR'                : r'sSFR',
    }

    return switcher.get(i, formatted_label(i, lumG=lumG))

def formatted_unit(i):
    """Return the unit to append after a numeric value."""

    switcher = {
        'Vdisp'               : r'km s$^{-1}$',
        'Radius_Bary_kpc'     : r'kpc',
        'size_Group_Bary_kpc' : r'kpc',
        'Radius_Circ_kpc'     : r'kpc',
        'size_Group_Circ_kpc' : r'kpc',
        'r_200'               : r'kpc',
        't_cr'                : r'Gyr',
        'Lum'                 : r'$L_\odot$',
        'GSL'                 : r'$L_\odot$',
        'GroupGSL'            : r'$L_\odot$',
        'Lum_BGG'             : r'$L_\odot$',
        'Lum_group'           : r'$L_\odot$',
        'Lum_Sat'             : r'$L_\odot$',
    }

    return switcher.get(i, "")

def formatted_sample_name(name):
    switcher={
        "CG4"       : r"CG$_4$",
        "Control4B" : r"Control$_{4B}$",
        "Control4C" : r"Control$_{4C}$",
        "RG4"       : r"RG$_4$"
    }
    return switcher.get(name,"Invalid sample name")

def display_label(value):
    """Return publication-facing labels while preserving internal category names."""

    switcher = {
        "Starforming": "Star-forming",
        "Star forming": "Star-forming",
        # Missing sSFR estimates are reported as counts, never as a class
        "NosSFR": "No sSFR",
        "Predom": "Predominant",
        "CG4_Gals": "CG4",
        "Control4B_Gals": "Control4B",
        "Control4C_Gals": "Control4C",
        "RG4_Gals": "RG4",
    }
    return switcher.get(value, value)

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
