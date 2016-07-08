import os

def rename():
	SRC = "../../../deepDB/srcPosters/"
	DST = "../../../deepDB/srcPosters/"
	# read file in this dir
	for i in range (400):
		try:
			file = SRC + `i` + '.jpg'
			newname = DST + `i`.zfill(3) + '.jpg'
			os.rename(file,newname)
			print (file)
		except IOError as e:
			print e

rename()