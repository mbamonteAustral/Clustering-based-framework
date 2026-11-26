# definitions and configuration values

import os
from sys import platform
import numpy as np


# ------ flags --------------
if platform == "linux" or platform == "linux2":
    show_plots = 0
elif platform == "win32":
    show_plots = 0 # if = 1, the script show all plots | Use show_plots = 0 when runing code with the VPN's PC.

# flags
fvsave_flag = 0  # save featurevextraction || = 1: yes, save features in csv file. 
preprocesssignal = 0 # if = 1, GSR and PPG will be preprocessed. 


# ------------------------------------------------------------------
# Cluster-Method configuration parameters:
# ------------------------------------------------------------------
number_of_clusters = int(os.environ.get("NUMBER_OF_CLUSTERS", "3"))
NUMBER_OF_BINS = 5 # Number of bins for mapping valence/arousal to 2D space (e.g., 4 or 9)
P_VALUE = 0.05


# ----- Valence / Arousal thresholds for mapping to X cells/bins in the 2D space
# This thresholds are used by the function map_label_to_cell to map 
# 9 bins thresholds
# valence/arousal values to a cell/bin in the 2D space

LOW_MED_THRESHOLD = 3.66
MED_HIGH_THRESHOLD = 6.33

if MED_HIGH_THRESHOLD < LOW_MED_THRESHOLD:
    raise ValueError(
        "MED_HIGH_THRESHOLD must be greater than LOW_MED_THRESHOLD"
    )

# 4 bins thresholds
MIDDLE_THRESHOLD = 5.0

# 5 bins thresholds
UPPER_MIDDLE_THRESHOLD = 5.5
MIDDLE_THRESHOLD = 5.0
LOWER_MIDDLE_THRESHOLD = 4.5

if MIDDLE_THRESHOLD < LOWER_MIDDLE_THRESHOLD or MIDDLE_THRESHOLD > UPPER_MIDDLE_THRESHOLD:
    raise ValueError(
        "MIDDLE_THRESHOLD must be between LOWER_MIDDLE_THRESHOLD and UPPER_MIDDLE_THRESHOLD"
    )

#----------------------------------------------------------------
# End of cluster-method configuration parameters
#----------------------------------------------------------------



# ------------- CONSTANTS --------------
# CASE dataset definitions

SAMPLING_RATE = 1000 # Hz  - biosignals sampling rate
ANNOTATION_SAMPLING_RATE = 20 # Hz
NEUTRAL_SCREEN = 1  # 1, corresponding to the blue screen/initial screen.
STIMULUS_SCREEN = 2  # 2, corresponding to a true stimulus.
VIDEO_ID_10 = 10
VIDEO_ID_11 = 11
VIDEO_ID_12 = 12

START_VIDEO_DURATION = 101500   # ms | Video-ID: 10
BASELINE_WINDOW = 60000 # ms 

COLUMNS = 1  # concatenate along columns 
INDEX = 0   # concatenate along index

# tripartite scheme
LOW_LABEL_MAP_VALUE = 0
MEDIUM_LABEL_MAP_VALUE = 11  # medium values ( Low < x < High  ) are mapped to a an arbitrary high number
HIGH_LABEL_MAP_VALUE = 1

# ---------------------------------------------------------------
# DEAP and K-EmoCon datasets' definitions
 
PPG_NUMBER_SAMPLES = 37440  # 585 sec of debate data (sampled at 64 Hz)
PPG_NUMBER_SAMPLES_PER_LABEL = 320  #5 sec (sampled at 64 hz)
NUMBER_LABELS = 117  # chunks of 5 sec debate data

GSR_NUMBER_SAMPLES = 2340  # 585 sec of debate data (sampled at 4 Hz)
GSR_NUMBER_SAMPLES_PER_LABEL = 20  #5 sec (sampled at 4 hz)


# GSR_NUMBER_LABELS = PPG_NUMBER_LABELS  # chunks of 5 sec debate data


# PPG_CHANNEL = 38   # measurement, left thumb
# GSR_CHANNEL = 36   # Ohms, sampling rate 128 Hz, Galvanic skin response, left middle and ring finger
# FP1_CHANNEL = 0 # see https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
# FP2_CHANNEL = 16 #            "
# F3_CHANNEL = 2 #            "
# F4_CHANNEL = 19  #            "


# Sample rate and desired bandpass cutoff frequencies (in Hz)
PPG_FS = 64.0  # Hz
GSR_FS = 4.0  # Hz
    
# GSR bandpass filter design parameters:
LOW_CUT_GSR = 0.07 # see Bach et al (0.01) | 0.07: avoids low freq component added by bandpass filter.
HIGH_CUT_GSR = 1.9 # See Singh et al | original = 2.1 | 1.9
ORDER_GSR_FILTER = 2  # higher order results in a unstable filter. | original = 3 | 
        
# PPG filter design parameters:
LOW_CUT_PPG = 0.05 #0.2  -- See Should PPG be filtered? section (notes)
HIGH_CUT_PPG = 18  #16 -- --- See Should PPG be filtered? section (notes)
ORDER_PPG_FILTER = 5 #bandpass butterflow filter order
    
# data augmentation constants:
SNR = 60 # [dB] - SNR - signal to noise ratio. A value of 32 dB preserves waveform. 
AUG_FACTOR = 1
    
# test dataset relative size (expressed in % of the dataset size)
TEST_DS_SIZE = 0.2

# save featurevextraction
FV_SAVE_FLAG = 0  # = 1: yes, save features in csv file. 
    
# NUM_VIDEOS = 40

SPLITTER_SEED = 12345 # with 6 good results for GBM (A: 0.75), knn (A: 0.75), RF (V: 0.75)
ESTIMATOR_SEED = 12345
    
# MEDIUM_LABEL_VALUES = np.nan  # medium values ( 3 < x < 7  ) are mapped to a an arbitrary high number (to be later discarted)
MEDIUM_LABEL_VALUES = 100  # medium values ( 3 < x < 7  ) are mapped to a an arbitrary high number (to be later discarted)

HIGH_LABEL_THRESHOLD = 5
LOW_LABEL_THRESHOLD = 5

