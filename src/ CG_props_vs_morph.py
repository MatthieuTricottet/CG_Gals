# region Imports

#* --------------------------------------------------------------------------------
#* General purpose imports
#* --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS

from scipy.stats import multivariate_normal, fisher_exact, barnard_exact
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines  # for legend proxies
import matplotlib.ticker as ticker

import seaborn as sns

import time 
import re 
from io import StringIO

#* --------------------------------------------------------------------------------
#* Specific plotting librairies imports
#* --------------------------------------------------------------------------------
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt



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

#* --------------------------------------------------------------------------------
#* Project functions imports
#* --------------------------------------------------------------------------------
import data_loader as dl
import generate_report as report
import sSFR

# endregion

