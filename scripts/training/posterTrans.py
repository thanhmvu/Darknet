import config as CFG
import cv2
import random
import imgTrans as iTr
import math
import numpy as np
# import utils

def addOcclusions(img):
	""" This method simulate the poster's audience as occlusions.
	The number of occlusions generated is random but in the range of [n1,n2].
	The width of all occlusions is the same, which is a forth of the poster's standard width (STD_SIZE/3) 
	
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
	Return the local 2D coordinates of each 3D points in set P, given that:
	- these points are on the same plane in the original 3D coordinate system
	- the size of P is greater or equal to 3
	
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
	R = random.randrange(int(w*0.75), int(w*1.25), unit)
	
	# Generate x
	xStart = xC - w/2 - R/math.sqrt(2)
	xEnd = xC + w/2 + R/math.sqrt(2)
	x = random.randrange(int(xStart), int(xEnd), unit)
	
	# Generate z
	dx = abs(xC - x) - w/2
	dz = math.sqrt(R*R - dx*dx)
	z = int(zC + dz) if dx > 0 else int(zC + R)
	
	# Generate y
	y = random.randrange(int(yC), int(yC + h), unit)
	
	return (x,y,z)


def perspective (img, title):
	h,w = img.shape[:2]
	[pt1,pt2,pt3,p4] = title
	C = [w*0.5, h*0.5, 0.0] # poster's center

	# Generate image plane 
	P = randViewpoint(C,w,h) # plane origin
	normal = np.subtract(C,P) # plane normal
	# Specify the viewpoint
	V = np.subtract(P, normal*4)
	
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
	title = dstPts2D.astype(int)

	return (ptImg,title)



# """ This method generates training images from the ground images using rotation """
# def rotate(img, angle):
# 	if img is None: 
# 		print 'ERROR: Input image is None'
# 		return None
# 	else:
# 		crop = False
# 		rotOut = rot.rotate(img,angle,crop)
# 		rtImg = rotOut[0]
		
# 		# compute the title box for training detection
# 		c = rotOut[1] # 4 corners of rotated poster
# 		h,w = rtImg.shape[:2]
# 		tBox = getTitleBox(c[0],c[1],c[2],c[3],w,h)
		
# # 		rtImg = drawTitleBox(rtImg,tBox)
# 		return (rtImg,tBox)


# """ This method generates training images from the ground images by rescaling them 
# scale the poster, not the image. scale down

# recalculate the tbox everytime
# """


# def scale(img, mult):	
# 	if img is None: 
# 		print 'ERROR: Input image is None'
# 		return None
# 	else:
# 		h,w = img.shape[:2]
# 		dim = ((int)(mult*w), (int)(mult*h)) # calculate new dimensions
# 		scImg = cv2.resize(img,dim, interpolation = cv2.INTER_LINEAR)
		
		
# 		return (scImg,tBox)
	
	