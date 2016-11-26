import random


# LIB_RANGE = (0,50) # the number of posters
num_classes = 10
POSTERS = sorted(random.sample(xrange(100), num_classes))
# LABELS = [i for i in range (LIB_RANGE[0], LIB_RANGE[1])] # list of class name
CLASSES = len(POSTERS)


NUM_VAR = 2000 # number of variation for each ground image
TASK = 'train'
# TASK = 'test'
NOTE = 'rand'


SRC_DIR = '../../../../database/realworld/set2/src/'
# DST_DIR = '../../../../database/realworld/set2/' +TASK+ '/' + `CLASSES` +'C_'+ `NUM_VAR` +'P_s'+ `LIB_RANGE[0]` +'_'+ NOTE +'/'
DST_DIR = '../../../../database/realworld/set2/%s/%dC_%dP_%s/' % (TASK,CLASSES,NUM_VAR,NOTE)

# SRC_DIR = '../../../../database/srcPosters/'
# DST_DIR = '../../../../database/'+ TASK +'Posters/'+ `CLASSES` +'C_'+ `NUM_VAR` +'P_s'+ `LIB_RANGE[0]` +'_'+ TASK +'/'

IMAGE_DIR = DST_DIR + 'JPEGImages/'
LABEL_DIR = DST_DIR + 'labels/'
BACKUP = DST_DIR + 'backup/'

STD_SIZE = 1000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

