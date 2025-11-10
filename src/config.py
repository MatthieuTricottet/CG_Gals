#* Display graphs or not while running the code
VERBOSE = True
SHOW = False
REBUILD_SAMPLE = False

#* File system
BASE_PATH = "/Users/matt/Astrophysics/CG_Gals/"
DATA_PATH = BASE_PATH + "data/"
OUTPUT_PATH = BASE_PATH + "output/"
CG_PATH = DATA_PATH + "CG_in_SDSSDR16/"
PROCESS_SAMPLES = "processed_sample.pkl"

RESULTS_BUILD = OUTPUT_PATH + "results_build.json"
RESULTS = OUTPUT_PATH + "results.json"

REPORT_FILE = "paper.tex"
REPORT_PATH = OUTPUT_PATH + "paper/"
SUBFIGURES_PATH = "figures/"
FIGURES_PATH = REPORT_PATH + SUBFIGURES_PATH
TEMPLATE_PATH = BASE_PATH + "src/Paper_template/"
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


sSFR_status = ['Quenched', 'Passive', 'Starforming']
Morphologies = ['Elliptical', 'Spiral', 'Uncertain']

sSFR_THRESHOLD, sSFR_QUENCHED = -1000, -14.0  # Log(sSFR) floor for quenched galaxies, in yr^-1

DOMINATIION_CRITERIA = 0.6

SAMPLE = {"CG4" : r"\CG","Control4B":r"\CB","Control4C":r"\CC","RG4":r"\RG"}
CONTROL = {samp : SAMPLE[samp] for samp in (s for s in SAMPLE if s not in ['CG4'])}
GASUFF = "_Gals"
GRSUFF = "_Groups"

