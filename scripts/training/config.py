
SRC_DIR = '../../../../db-deepnet/srcPosters/'
# DST_DIR = '../../../../db-deepnet/training/'
# LABELS_DIR = '../../../../db-deepnet/labels/'
DST_DIR = './images/'
LABELS_DIR = './labels/'

LIB_SIZE = 2 # the total number is 400 ground images
NUM_VAR = 2 # number of variation for each ground image
STD_SIZE = 1000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

# PT_RANGE = range(3,31,27) # range(3,31,3)
# RT_RANGE = range(4,41,36) # range(4,41,4)
# SC_RANGE = range(1,22,6) # range(1,22,1)
# # BL_RANGE = range(0,1,1)
# # TL_RANGE = range(0,1,1)