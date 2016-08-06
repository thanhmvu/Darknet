
SRC_DIR = '../../../../database/srcPosters/'
DST_DIR = '../../../../database/testPosters/5C_100P_s0_test/JPEGImages/'
LABELS_DIR = '../../../../database/testPosters/5C_100P_s0_test/labels/'
NOTE_DIR = '../../../../database/testPosters/5C_100P_s0_test/'

LIB_RANGE = (0,5) # the total number is 400 ground images
NUM_VAR = 100 # number of variation for each ground image
STD_SIZE = 2000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

