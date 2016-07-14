import cv2

# Resize the image to have the width of given size 
# preserve the ratios between 2 side 
def resize(img,size): 
  h,w = img.shape[:2]
  ratio = float(h)/w
  dim = (size, int(size*ratio)) # calculate new dimensions (w,h)
  return cv2.resize(img,dim)


# Method to visualize the title box by drawing the bounding box of the title and its center onto given image
# tBox = ((xCenter/imgW), (yCenter/imgH), (objW/imgW), (objH/imgH))
def drawTitleBox(img,tBox):
	# Extract the coordinates
	h,w = img.shape[:2]
	x = int(tBox[0] * w) # x-center of the title box
	y = int(tBox[1] * h) # y-center
	wB = int(tBox[2] * w) # the width of the box
	hB = int(tBox[3] * h) # the height
	
	# Calculate the position of top left corner
	x1 = x - wB/2; x2 = x1 + wB
	y1 = y - hB/2; y2 = y1 + hB
	
	# Draw the box and the center
	cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),3) 
	cv2.circle(img,(x,y),10,(0,0,255),3)
	return img