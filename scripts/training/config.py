
SRC_DIR = '../../../../database/srcPosters/'
DST_DIR = '../../../../database/testPosters/0_90_100/JPEGImages/'
LABELS_DIR = '../../../../database/testPosters/0_90_100/labels/'
NOTE_DIR = '../../../../database/testPosters/0_90_100/'

LIB_RANGE = (0,90) # the total number is 400 ground images
NUM_VAR = 100 # number of variation for each ground image
STD_SIZE = 2000 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height

OCC_W = STD_SIZE/4
OCC_NUM = (0,10) # randint between this range

