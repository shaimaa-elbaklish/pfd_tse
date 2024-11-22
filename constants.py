import pytz


###########################################################################
# GENERAL CONSTANTS
###########################################################################
METER2FT = 3.28084
SEC2HR = 1/3600.0
HR2SEC = 3600.0
FT2MILE = 1/5280.0
MILE2FT = 5280.0

FREE_FLOW_SPEED = 43.0
BACKWARDS_WAVE_SPEED = -13.0

PFD_MIN_OBSERVATIONS = 15
PFD_MAX_DENSITY= 120

###########################################################################
# CONSTANTS FOR NGSIM
###########################################################################
NGSIM_TIMEZONE = pytz.timezone("US/Pacific")
NGSIM_DATA_ROOT = "C:/Users/selbaklish/Desktop/Python_Workspace/NGSIM-data"
NGSIM_MIN_POSITION = 50.0
NGSIM_TESTBED_LENGTH = 1950.0
NGSIM_MAX_SPEED = 65.0
NGSIM_FREQUENCY = 10.0


###########################################################################
# CONSTANTS FOR I24-MOTION
###########################################################################
I24_TIMEZONE = pytz.timezone("US/Central")
I24_DATA_ROOT = "C:/Users/selbaklish/Desktop/Python_Workspace/PFD/results/I24Motion/extracted_data"
I24_MIN_MILEMARKER = 58.7 * MILE2FT
I24_MIN_POSITION = 0.0 * MILE2FT
I24_TESTBED_LENGTH = 4.2 * MILE2FT
I24_MAX_SPEED = 80.0
I24_FREQUENCY = 25.0
