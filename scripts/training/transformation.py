import config as CFG
import cv2
import random

""" This method simulate the poster's audience as occlusions.
The number of occlusions generated is random but in the range of [0,10].
The width of all occlusions is the same, which is a forth of the poster's standard width (STD_SIZE/3) """
def addOcclusions(img):
	occWidth = CFG.OCC_W
	n1,n2 = CFG.OCC_NUM
	numOfOcc = random.randint(n1,n2)
  
	h,w = img.shape[:2]
	# (x-start, x-end, x-step). same for ys 
	# The step is to make sure the occlusions are spaced out reasonably
	xs = (0, w-1, occWidth/4) 
	ys = (int(h* CFG.TITLE_RATIO), int(h*0.5), int(h*0.1)) 
	
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
	
	return img 

	
""" This method generates training images from the ground images using perspective transformation.
An assumption is that the input poster is a rectangle
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
		height = float(maxY - minY + 1) / ptImg.shape[0] # obj_h / img_h
		
		tBox = (x, y, width, height)
		
# 		ptImg = drawTitleBox(ptImg,tBox)
		return (ptImg,tBox)

""" Help method for perpspectiveTranasform.
Prints out the tuple of perpspective ratios as string """
def ratiosToStr(tupleR):
	out = ''
	for r in tupleR:
		if r == 0: out +="x"
		else: out += `int(r*100)`.zfill(2)
	return out


""" 
The corners of the title:
    TopLeft: 1, TopRight: 2
    BottomL: 3, BottomR:  4
Input c1 - c4 are the 4 corners of the poster image
imgW, imgH are the dimensions of that poster
"""
def getTitleBox(c1,c2,c3,c4,imgW,imgH):
	# Get the title corners
	tc1 = c1 ; tc2 = c2
	tc3 = tuple(c1[i] + int((c3[i] - c1[i]) * TITLE_RATIO) for i in [0,1])
	tc4 = tuple(c2[i] + int((c4[i] - c2[i]) * TITLE_RATIO) for i in [0,1])
	
	# Get bounding box
	minX = min(tc1[0], tc2[0], tc3[0], tc4[0])
	maxX = max(tc1[0], tc2[0], tc3[0], tc4[0])
	minY = min(tc1[1], tc2[1], tc3[1], tc4[1])
	maxY = max(tc1[1], tc2[1], tc3[1], tc4[1])
  
	# Calculate width, height, and center's coordinates
	wTitle = maxX - minX + 1
	hTille = maxY - minY + 1
	xCenter = (maxX + minX) /2
	yCenter = (maxY + minY) /2
	
	# Return the tile box's values relative to the containing image's dimensions
	x = float(xCenter) / imgW
	y = float(yCenter) / imgH
	w = float(wTitle) / imgW
	h = float(hTille) / imgH
	
	return (x, y, w, h)


""" This method generates training images from the ground images using rotation """
def rotate(img, angle):
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		crop = False
		rotOut = rot.rotate(img,angle,crop)
		rtImg = rotOut[0]
		
		# compute the title box for training detection
		c = rotOut[1] # 4 corners of rotated poster
		h,w = rtImg.shape[:2]
		tBox = getTitleBox(c[0],c[1],c[2],c[3],w,h)
		
# 		rtImg = drawTitleBox(rtImg,tBox)
		return (rtImg,tBox)


""" This method generates training images from the ground images by rescaling them 
scale the poster, not the image. scale down

recalculate the tbox everytime
"""


def scale(img, mult):	
	if img is None: 
		print 'ERROR: Input image is None'
		return None
	else:
		h,w = img.shape[:2]
		dim = ((int)(mult*w), (int)(mult*h)) # calculate new dimensions
		scImg = cv2.resize(img,dim, interpolation = cv2.INTER_LINEAR)
		
		
		return (scImg,tBox)
	
	