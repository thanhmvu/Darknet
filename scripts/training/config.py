
LIB_RANGE = (0,10) # the total number is 400 ground images
NUM_VAR = 100 # number of variation for each ground image
# TASK = 'train'
TASK = 'test'



SRC_DIR = '../../../../database/srcPosters/'

classes = LIB_RANGE[1] - LIB_RANGE[0]
DST_DIR = '../../../../database/'+ TASK +'Posters/'+ `classes` +'C_'+ `NUM_VAR` +'P_s0_'+ TASK +'/'
IMAGE_DIR = DST_DIR + 'JPEGImages/'
LABEL_DIR = DST_DIR + 'labels/'
BACKUP = DST_DIR + 'backup/'

STD_SIZE = 2000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

