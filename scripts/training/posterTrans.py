import config as CFG
import cv2
import random
import imgTrans as iTr
import math
import numpy as np
import utils

def addOcclusions(img):
	""" This method simulate the poster's audience as occlusions.
	The number of occlusions generated is random but in the range of [n1,n2].
	The width of all occlusions is the same, which is a forth of the poster's standard width (STD_SIZE/3) 
	
	@param img - the image to add occlusions
	@return the image with occlusions on it
	
	"""
	occWidth = CFG.OCC_W
	n1,n2 = CFG.OCC_NUM
	numOfOcc = random.randint(n1,n2)
  
	h,w = img.shape[:2]
	# (x-start, x-end, x-step). same for ys 
	# The step is to make sure the occlusions are spaced out reasonably
	xs = (0, w-1, occWidth/4) 
	ys = (int(h* CFG.TITLE_RATIO), int(h*0.5), int(h*0.05)) 
	
	# Generate occlusions
	for it in range(0,numOfOcc):
		# Generate top point
		x1 = random.randrange(xs[0], xs[1], xs[2])		
		y1 = random.randrange(ys[0], ys[1], ys[2])
		pt1 = (x1,y1)
		# Calculate end point
		pt2 = (x1 + occWidth, h-1)
		# Add occlusion
		cv2.rectangle(img,pt1,pt2,(150,150,150),-1)
	
	return img 


""" ======================================== Perspective ======================================== """

def localCoords(P):
	""" Adopted and modified from http://stackoverflow.com/questions/26369618/getting-local-2d-coordinates-of-vertices-of-a-planar-polygon-in-3d-space
	
	@param P - a set of 3 or more 3D points on the same plane P
	@return a set of local 2D coordinates of all 3D points relative to plane P, with the origin being the first point in the given set P
	
	"""
	loc0 = P[0]                      # local origin
	locy = np.subtract(P[len(P)-1],loc0)  	 # local Y axis
	normal = np.cross(locy, np.subtract(P[1],loc0)) # a vector orthogonal to polygon plane
	locx = np.cross(normal, locy)      # local X axis

	# normalize
	locx /= np.linalg.norm(locx)
	locy /= np.linalg.norm(locy)

	local_coords = [(np.dot(np.subtract(p,loc0), locx),  # local X coordinate
	                 np.dot(np.subtract(p,loc0), locy))  # local Y coordinate
	                for p in P]
	
	# Handle cropping issue
	dx = -min([p[0] for p in local_coords])
	dy = -min([p[1] for p in local_coords])
	local_coords = [(p[0]+dx, p[1]+dy) for p in local_coords]

	return local_coords


def randViewpoint(C,w,h):
	""" Method that generates a point for the perpsective plane (the normal of this plane would be toward the center C of the poster)
	The point is generates randomly within a hardcoded range, relative to C,w,h. The range is illustrated below.
	
			      |_poster_|           .      a = 45 degree
			    / |        | \         |      zRange = (zC + R/2) +-  R/2
			  / a |        | a \       | R    xRange =     xC     -+ (w/2 + R.sin(a))
			/     |        |     \     |      yRange = (yC - h/2) +-  h/2
			\_____|________|_____/     |	
			       --------
			          w
	
	@param C - the center of the poster
	@param w - the width of the poster
	@param h - the height of the poster
	@return (x,y,z) - the coordinates of the generated point
	
	"""
	(xC,yC,zC) = C # poster's center
	unit = 1 # estimated to be roughly equivalent to 1/2 -> 1 inch in real world unit
	R = random.randrange(int(w*0.25), int(w*0.75), unit)
	
	# Generate x
	xStart = xC - w/2 - R/math.sqrt(2)
	xEnd = xC + w/2 + R/math.sqrt(2)
	x = random.randrange(int(xStart), int(xEnd), unit)
	
	# Generate z
	dx = abs(xC - x) - w/2
	z = int(zC + R) if dx < 0 else int(zC + math.sqrt(R*R - dx*dx)) 	
	
	# Generate y
	y = random.randrange(int(yC-h/2), int(yC), unit)
	
	return (x,y,z)


def cropPtImg(ptImg, M):
	""" Method that crops the redundant background from a perspective transformed image
	
	@param ptImg - the transformed image to be cropped
	@param M - the 4x4 perspective matrix
	@return the cropped image
	
	"""
	h,w = ptImg.shape[:2]
	oldCorners = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
	
	# Calculate new corners
	newCorners = [utils.transform2D(pt,M) for pt in oldCorners]
	minX,maxX,minY,maxY = utils.boundingArea(newCorners)
	
	# Check if the transformed corners are out of bound
	x1 = max(minX,0)
	x2 = min(maxX,w)
	y1 = max(minY,0)
	y2 = min(maxY,h)
	
	return ptImg[y1:y2,x1:x2]


def perspective (img, title):
	""" This method generates training images from the ground images using perspective transformation.
	
	@param title - list of 4 corners of the title
	@param img - the image to be transformed
	@return (imgT,title) - the transformed image and coordinates of the title' corners
	
	"""
	h,w = img.shape[:2]
	[pt1,pt2,pt3,p4] = title
	C = [w*0.5, h*0.5, 0.0] # poster's center

	# Generate image plane 
	P = randViewpoint(C,w,h) # plane origin
	normal = np.subtract(C,P) # plane normal
	# Hardcoded viewpoint
	r = random.randint(75,125)*0.01
	V = np.subtract(P, normal*r)
	
	M1 = iTr.projection_matrix(P, normal, None, V) # 4x4 matrix

	# Convert to 3x3 2D matrix
	dstPts3D = []
	for pt in title:
		homoPt = np.array([[pt[0]], [pt[1]], [0], [1]])
		out = np.dot(M1,homoPt)
		dstPt = [out[0][0]/out[3][0],out[1][0]/out[3][0],out[2][0]/out[3][0]]
		dstPts3D.append(dstPt)

	# Convert 3D coords to 2D coords
	dstPts2D = np.float32(localCoords(dstPts3D))
	srcPts2D = np.float32(title)

	# get 3x3 pespsective matrix
	M2 = cv2.getPerspectiveTransform(srcPts2D, dstPts2D)

	# Transform
	ptImg = cv2.warpPerspective(img,M2,(w,h))
	
	# Refine the results
	title = dstPts2D.astype(int)
	ptImg = cropPtImg(ptImg,M2)

	return (ptImg,title)


""" ======================================== Rotation ======================================== """

def rotatePoint(pt, mat):
	""" Method to rotate a 2D point using a given matrix
	
	@param pt - the point to rotate
	@param mat - the 3x3 rotation matrix
	@return (x,y) - the new coordinates of the point after applying the matrix
	
	"""
	tmp = np.array([[pt[0]],[pt[1]],[1]])
	ptMat = np.dot(mat, tmp)
	x = int(ptMat[0][0])
	y = int(ptMat[1][0])
	return (x,y)

def rotate(img, title):
	""" This method generates training images from the ground images using rotation.
	
	@param title - list of 4 corners of the title
	@param img - the image to be transformed
	@return (imgT,title) - the transformed image and coordinates of the title' corners
	
	"""
	angle = random.randint(-30,30)
	h,w = img.shape[:2]
	poster = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)] # 4 corners
	
	# Copy image onto a bigger background to avoid cropping when being rotated
	R = max(w,h)*2 
	imgB = np.zeros((R,R,3), np.uint8) # create a black backgrouund
	y = R/2-h/2 ; x = R/2-w/2
	imgB[y: y+h, x: x+w] = img # Copy img onto the background
	hB,wB = imgB.shape[:2]
	poster = [(p[0] -w/2 +wB/2, p[1] -h/2 +hB/2) for p in poster]
	title = [(p[0] -w/2 +wB/2, p[1] -h/2 +hB/2) for p in title] 

	# Calculate the rotation matrix then apply it to the image
	M = cv2.getRotationMatrix2D((wB/2,hB/2), angle, 1) # (center(x,y),angle,scale)
	
	rtImg = cv2.warpAffine(imgB,M,(wB,hB))
	poster = [rotatePoint(pt,M) for pt in poster]
	title = [rotatePoint(pt,M) for pt in title]
	hR,wR = rtImg.shape[:2]
	
	# Crop redundant background
	minX,maxX,minY,maxY = utils.boundingArea(poster)
	# check if the transformed corners are out of bound
	x1 = max(minX,0) ; x2 = min(maxX,wR)
	y1 = max(minY,0) ; y2 = min(maxY,hR)
	rtImg = rtImg[y1:y2,x1:x2] # crop
	hC,wC = rtImg.shape[:2]
	title = [(p[0] -wR/2 +wC/2, p[1] -hR/2 +hC/2) for p in title] 
	
	return (rtImg,title)

""" ======================================== Scaling ======================================== """


def scale(img, title):	
	""" This method generates training images from the ground images by rescaling the poster inside those images
	
	@param title - list of 4 corners of the title
	@param img - the image to be transformed
	@return (imgT,title) - the transformed image and coordinates of the title' corners
	
	"""
	scImg = img
# 	if img is None: 
# 		print 'ERROR: Input image is None'
# 		return None
# 	else:
# 		h,w = img.shape[:2]
# 		dim = ((int)(mult*w), (int)(mult*h)) # calculate new dimensions
# 		scImg = cv2.resize(img,dim, interpolation = cv2.INTER_LINEAR)
		
	return (scImg,tBox)
	
	