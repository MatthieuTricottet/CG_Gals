#* Display graphs or not while running the code
VERBOSE = True
SHOW = False
REBUILD_SAMPLE = False
# Render the paper from existing JSON outputs without rerunning analyses.
RENDER_PAPER_ONLY = True

#* File system
BASE_PATH = "/Users/matt/Astrophysics/CG_Gals/"
DATA_PATH = BASE_PATH + "data/"
OUTPUT_PATH = BASE_PATH + "output/"
CG_PATH = DATA_PATH + "CG_in_SDSSDR16/"
PROCESS_SAMPLES = "processed_sample.pkl"

RESULTS_BUILD = OUTPUT_PATH + "results_build.json"
RESULTS = OUTPUT_PATH + "results.json"

#* Galaxy-size data products (cached external fetches, see src/size_data.py)
SIZE_COLUMNS_FILE = DATA_PATH + "sdss_size_columns.csv"
SIMARD_SUBSET_FILE = DATA_PATH + "simard2011_subset.csv"
SIMARD_FTP_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/196/11/"

REPORT_FILE = "paper.tex"
REPORT_PATH = OUTPUT_PATH + "paper/"
SUBFIGURES_PATH = "figures/"
FIGURES_PATH = REPORT_PATH + SUBFIGURES_PATH
TEMPLATE_PATH = BASE_PATH + "src/paper_template/"
TEMPLATE_FILE = "paper_template.tex" 
BIB_FILE = "paper"


#* --------------------------------------------------------------------------------
#* Constants
#* --------------------------------------------------------------------------------

#* Limits for completeness
Z_MIN = 0.005
Z_MAX = 0.0452
R_MAX = 17.77

#* Specify the SDSS data release 
DATA_RELEASE = 16


# Measured sSFR classes (GMM separation in the logM*-log sSFR plane).
# Galaxies without an sSFR measurement carry sSFR_status = NosSFR_LABEL and
# are excluded from the classification, from every sSFR figure, and from all
# quenched/star-forming fractions; they are reported as counts only.
sSFR_status = ['Quenched', 'Starforming']
NosSFR_LABEL = 'NosSFR'
Morphologies = ['Elliptical', 'Spiral', 'Uncertain']

# Sanity range for measured log sSFR (yr^-1) and log stellar mass: values
# outside are treated as unmeasured (legacy catalogues used -9999 sentinels).
sSFR_VALID_RANGE = (-25.0, -5.0)
LGM_VALID_RANGE = (5.0, 14.0)

DOMINATIION_CRITERIA = 0.6

SAMPLE = {"CG4" : r"\CG","Control4B":r"\CB","Control4C":r"\CC","RG4":r"\RG"}
CONTROL = {samp : SAMPLE[samp] for samp in (s for s in SAMPLE if s not in ['CG4'])}
GASUFF = "_Gals"
GRSUFF = "_Groups"

# Limit for statistical significance
P_LIMIT = 5e-2

#* Galaxy-size quality cuts (Planck15 kpc after the Scale->arcsec re-conversion)
SIZE_MIN_KPC = 0.1  # smallest physically credible half-light radius
SIZE_MAX_KPC = 50.0  # largest physically credible half-light radius
# Sersic indices at the GIM2D fit bounds are flagged as pegged and excluded
# from the primary Simard size sample (kept for a robustness variant).
NG_PEG_LOW = 0.55
NG_PEG_HIGH = 7.9
