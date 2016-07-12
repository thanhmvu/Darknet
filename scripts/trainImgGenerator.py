import cv2
import numpy as np
import os
import random

""" ============================================================================================================
This script generates a set of images for training darknet from the ground database of 400 posters.
For each transformation methods, keep in mind that the deep net should use the title of the posters to recognize the whole poster.

The title of a poster is represent using the following parameters:
    <x> <y> <width> <height>
Where x, y, width, and height are relative to the image's width and height, with x and y are the location of the center of the object:
    x = x_center / img_w
    y = y_center / img_h
    with = obj_w / img w
    height = obj_h / img_h
============================================================================================================ """

SRC_DIR = '../../../db-deepnet/srcPosters/'
DST_DIR = '../../../db-deepnet/training/'
if not os.path.exists(DST_DIR): os.mkdir(DST_DIR)
LABELS_DIR = '../../../db-deepnet/labels/'
if not os.path.exists(LABELS_DIR): os.mkdir(LABELS_DIR)

NUM_OF_IMG = 2 # the total number is 400 ground images
STD_SIZE = 500 
TITLE_RATIO = 0.2 # is estimated to be the ratio of title's height/ poster's height
PT_RANGE = range(3,31,27) # range(3,31,3)

""" This method makes all image to have the same width as a way of standardizing their sizes """
def sizeStandardize(img,std_width): 
  height, width = img.shape[:2]
  ratio = float(height)/width  
  dim = (std_width, (int)(std_width*ratio)) # calculate new dimensions
  res = cv2.resize(img,dim)  
  return res


""" This method simulate the poster's audience as occlusions
The number of occlusions generated is random but in the range of [0,10]
The width of all occlusions is the same, which is a third of the poster's standard width (STD_SIZE/3) """
def addOcclusions(img):
	h,w = img.shape[:2]
	
	numOfOcc = random.randint(0,10)
	occWidth = STD_SIZE/3
	# (x-start, x-end, x-step). same for ys 
	# The step is to make sure the occlusions are spaced out reasonably
	xs = (0, w-1, occWidth/4) 
	ys = (int(h*TITLE_RATIO), int(h*0.5), int(h*0.1)) 
	
	# Generate occlusions
	for it in range(0,numOfOcc):
		# Generate top point
		x1 = random.randrange(xs[0], xs[1], xs[2])		
		y1 = random.randrange(ys[0], ys[1], ys[2])
		pt1 = (x1,y1)
		# Calculate end point
		pt2 = (x1 + occWidth, h-1)
		# Add occlusion
		cv2.rectangle(img,pt1,pt2,(0,0,0),-1)
	
# 	return img 


def resize(img, path_out, scale):	
	if img is None: 
		print 'ERROR: Cannot read' + path_in
		return False
	else:
		height, width = img.shape[:2]
		dim = ((int)(scale*width), (int)(scale*height)) # calculate new dimensions
		res = cv2.resize(img,dim, interpolation = cv2.INTER_LINEAR)
		cv2.imwrite(path_out,res) # save image
		return True
	
	
def rotate(img, path_out, angle, scale):
	if img is None: 
		print 'ERROR: Cannot read' + path_in
		return False
	else:
		rows,cols = img.shape[:2]
		M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale) # (center,angle,scale)
		dst = cv2.warpAffine(img,M,(cols,rows))
		cv2.imwrite(path_out,dst)
		return True

	
""" This method generates training images from the ground images using perspective transformation 
The input ratios r1-r4 are used to generate the 4 corners of new image in the following way:
	top-l-point: (0, h * r1)     , top-r-point: (w, h * r2),
	bottom-l-pt: (0, h - h * r2) , bottom-r-pt: (w, h - h * r4) 
"""
def perspectiveTransform (img, (r1,r2,r3,r4)):
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		h,w = img.shape[:2]

		# the original 4 points (corners) of the image to transform
		src_points = np.float32([ [0,0], [w,0],
		                          [0,h], [w,h] ])
		# the corresponding new 4 location to transform original points to
		y1 = 0 + int(h*r1) ; y2 = 0 + int(h*r2)
		y3 = h - int(h*r3) ; y4 = h - int(h*r4)
		dst_points = np.float32([ [0,y1], [w,y2],
		                          [0,y3], [w,y4] ])
		
		# compute the transform matrix and apply it
		M = cv2.getPerspectiveTransform(src_points,dst_points)
		ptImg = cv2.warpPerspective(img,M,(w,h))
		
		# compute the title box for training detection
		minY = min(y1,y2)
		maxY1 = y1 + int((y3-y1)*TITLE_RATIO)
		maxY2 = y2 + int((y4-y2)*TITLE_RATIO)
		maxY = max(maxY1,maxY2) # the lowest corner of the title area
		
		x = 0.5 # x_center / img_w (since the title's width = the poster's width)
		y = (minY + maxY)*0.5 / ptImg.shape[0] # y_center / img_h
		width = 1 # obj_w / img w (since the title's width = the poster's width)
		height = float(maxY - minY) / ptImg.shape[0] # obj_h / img_h
		
		tBox = (x, y, width, height)
		
		return (ptImg,tBox)
	
def ratiosToStr(tupleR):
	out = ''
	for r in tupleR:
		if r == 0: out +="x"
		else: out += `int(r*100)`.zfill(2)
	return out
	
""" ======================================== Begining of main code ======================================== """
cnt = 0
for i in range (0,NUM_OF_IMG):
	# read the image from the source folder
	path_in = SRC_DIR + `i`.zfill(3) + '.jpg'
	img = cv2.imread(path_in)
	
	if img is None: 
		print 'ERROR: Cannot read' + path_in
	else:
		img = sizeStandardize(img,STD_SIZE)
		addOcclusions(img)
		
# 		path_out = DST_DIR+`cnt`.zfill(6)+'.jpg'
# 		cv2.imwrite(path_out,img)
# 		cnt += 1
# 		print "Created", path_out
		
		### Perspective Transformation
		# generate the ratios of left and right edges
		ptRange = [x*0.01 for x in PT_RANGE]
		rightRatios = [(r,0,r,0) for r in ptRange]
		leftRatios  = [(0,r,0,r) for r in ptRange]
		ratios = rightRatios + leftRatios
		
		for rs  in ratios:
			ptOut = perspectiveTransform(img,rs)	
			if ptOut != None:
				imgT = ptOut[0]
				titleBox = ptOut[1]
				
				## Save the training image
				# name format: groundIdx_trans_transType_trainIdx.jpg
				img_out = DST_DIR + `cnt`.zfill(6) +'_'+ `i`.zfill(3) +'_ptrans_'+ ratiosToStr(rs)  +'.jpg'
				cv2.imwrite(img_out,imgT)
				print 'Created ' + img_out
				
				## Save the label file
				label_out = LABELS_DIR + `cnt`.zfill(6) +'.txt'
				f = open(label_out,'w')
				line = `i` + ' ' + `titleBox`.strip('()').replace(',','')
				f.write(line)
				f.close()
				print 'Created ' + label_out
				
			else:
				print 'Fail to create' + path_out
				
			cnt += 1

	
		
# 		### Rotation
# 		for angle  in [-30, -20, -10, 10, 20, 30]:
# 			scale = 1
# 			# create the directory
# 			dir = query_path+'rotation_'+`angle`+'/'
# 			if not os.path.exists(dir): os.mkdir(dir)	
# 			path_out = dir+`i`+'.jpg'
# 			# rotate and save new image
# 			if rotate (img, path_out, angle, scale):
# 				print 'Created ' + path_out
# 			else:
# 				print 'Fail to create' + path_out
		
# 		### Scaling
# 		for scale in [0.2, 0.5, 1.5, 2]:
# 			# create the directory
# 			dir = query_path+'scaling_'+`scale`+'/'
# 			if not os.path.exists(dir): os.mkdir(dir)	
# 			path_out = dir+`i`+'.jpg'

# 			# resize and save new image
# 			if resize (img, path_out, scale):
# 				print 'Created ' + path_out
# 			else:
# 				print 'Fail to create' + path_out
				
# 		### Translation
		
# 		### Blurring

