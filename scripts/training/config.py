
LIB_RANGE = (0,20) # the total number is 400 ground images
NUM_VAR = 1000 # number of variation for each ground image
TASK = 'train'
# TASK = 'test'
NOTE = 'light'

CLASSES = LIB_RANGE[1] - LIB_RANGE[0]
LABELS = [i for i in range (LIB_RANGE[0], LIB_RANGE[1])] # list of class name

SRC_DIR = '../../../../database/realworld/set2/src/'
DST_DIR = '../../../../database/realworld/set2/' +TASK+ '/' + `CLASSES` +'C_'+ `NUM_VAR` +'P_s'+ `LIB_RANGE[0]` +'_'+ NOTE +'/'

# SRC_DIR = '../../../../database/srcPosters/'
# DST_DIR = '../../../../database/'+ TASK +'Posters/'+ `CLASSES` +'C_'+ `NUM_VAR` +'P_s'+ `LIB_RANGE[0]` +'_'+ TASK +'/'

IMAGE_DIR = DST_DIR + 'JPEGImages/'
LABEL_DIR = DST_DIR + 'labels/'
BACKUP = DST_DIR + 'backup/'

STD_SIZE = 1000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

