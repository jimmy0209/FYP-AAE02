
#------------------------------------#
# Author: Yueh-Lin Tsou              #
# Update: 7/14/2019                  #
# E-mail: hank630280888@gmail.com    #
#------------------------------------#

"""------------------------------------
Canny Edge Detection Implementation
-------------------------------------"""

import pylab as pl	# imoprt matplotlib's subpackage as pl use for graph
import numpy as np	# use numpy library as np for array object
import cv2			# opencv-python
import scipy		# call math function, [scipy.hypot, scipy.arctan]
import math			# call math function, [math.pi]
import argparse

# ------------------- Function to do Gaussian Filtering to reduce noise ------------------- #
def Image_Filtering(img):
    Gimg = cv2.GaussianBlur(grayimg,(3, 3),0)
    return Gimg

# ------------------- Function to do image padding ------------------- #
def Image_Padding(Gimg):
    Gnonimg = np.zeros((x, y), dtype = 'i')   # copy Gaussian image "Gimg" to "Gnonimg" inorder to avoid data overflow in image array
    for i in range(0, x):					  # for-loop from row 0 to x
        for j in range(0, y):				  # for-loop from column 0 to y
            Gnonimg[i, j] = Gimg[i, j]		  # copy the image values

    return Gnonimg

# ------------------- Function to find intendity and orientaion ------------------- #
def Intensity_and_Orientation(Gnonimg):
    ###	set first derivative in horizontal and vertical orientation. Find magnitude and orientation of gradient for each pixel
    GX = np.ones((x, y), dtype = 'f')			# first derivative in horizontal orientation
    GY = np.zeros((x, y), dtype = 'f')			# first derivative in vertical orientation
    magnitude = np.zeros((x, y), dtype = 'f')	# magnitude of gradient
    orientation = np.zeros((x, y), dtype = 'f')	# orientation of gradient

    """
    ### simple filter
    for i in range(1, x-1): 	# set first derivative from 1 to x-1 (because of the edge of the image)
        for j in range(1, y-1): # set first derivative from 1 to y-1 (because of the edge of the image)
    		GX[i, j]=(Gnonimg[i, j+1]-Gnonimg[i, j]+Gnonimg[i+1, j+1]-Gnonimg[i+1, j])  # simple filter in x diection
    		GY[i, j]=(Gnonimg[i+1, j]-Gnonimg[i, j]+Gnonimg[i+1, j+1]-Gnonimg[i, j+1])  # simple filter in y diection
    """

    ### Sobel filter
    for i in range(1, x-1): 	# set first derivative from 1 to x-1(because of the edge of the image)
        for j in range(1, y-1): # set first derivative from 1 to y-1(because of the edge of the image)
            GX[i, j]=((Gnonimg[i-1, j+1]-Gnonimg[i-1, j-1]+2*(Gnonimg[i, j+1]-Gnonimg[i, j-1])+Gnonimg[i+1, j+1]-Gnonimg[i+1, j-1])) # sobel filter in X diection
            GY[i, j]=((Gnonimg[i+1, j-1]-Gnonimg[i-1, j-1]+2*(Gnonimg[i+1, j]-Gnonimg[i-1, j])+Gnonimg[i+1, j+1]-Gnonimg[i-1, j+1])) # sobel filter in Y diection


    magnitude = scipy.hypot(GX, GY) # calculate magnitude value of each pixel

    # if GX == 0 then GX = 1, in order to avoid error when calculate orientation value
    for i in range(1, x-1):
        for j in range(1, y-1):
            if GX[i,j]==0:
               GX[i,j]=1

    orientation = scipy.arctan(GY/GX) # calculate orientation value of each pixel

    ### transform orientation value to degree (orientation*180/pi), then clasify each pixel into 0, 45, 90 and 135 degree
    for i in range(0, x):		# count pixel from 0 to x
        for j in range(0, y):	# count pixel from 0 to y
            orientation[i, j] = orientation[i, j]*180/math.pi # transform orientation into degree
            if orientation[i, j]<0: # tranform which degree < 0 to 0-360
                orientation[i, j] = orientation[i, j]+360 # if degree is negative +360 to become positive degree

    		# classify every pixel
            if (orientation[i, j]<22.5 and orientation[i, j]>=0) or (orientation[i, j]>=157.5 and orientation[i, j]<202.5) or (orientation[i, j]>=337.5 and orientation[i, j]<=360):
                   orientation[i, j]=0 # if 0<=degree<225 or 157.5<=degree<202.5 or 337.5<=degree<360 the pixel orientation = 0
            elif (orientation[i, j]>=22.5 and orientation[i, j]<67.5) or (orientation[i, j]>=202.5 and orientation[i, j]<247.5):
                   orientation[i, j]=45 # if 22.5<=degree<67.5 or 202.5<=degree<247.5 the pixel orientation = 45
            elif (orientation[i, j]>=67.5 and orientation[i, j]<112.5)or (orientation[i, j]>=247.5 and orientation[i, j]<292.5):
                   orientation[i, j]=90 # if 67.5<=degree<112.5 or 247.5<=degree<292.5 the pixel orientation = 90
            else:
                   orientation[i, j]=135 # if 112.5<=degree<157.5 or 292.5<=degree<337.5.5 the pixel orientation = 135

    return  magnitude, orientation

# ------------------- Function to do Non-maximum Suppression ------------------- #
def Suppression(magnitude, orientation):
    for i in range(1, x-1):		# count pixel from 1 to x-1
        for j in range(1, y-1): # count pixel from 1 to y-1
            if orientation[i,j]==0: # if the pixel orientation = 0, compare with it's right and left pixel
                if (magnitude[i, j]<=magnitude[i, j+1]) or (magnitude[i, j]<=magnitude[i, j-1]): # if these pixel's magnitude are all bigger than magnitude[i,j]
                    magnitude[i][j]=0 # set magnitude[i, j]=0
            elif orientation[i, j]==45: # if the pixel orientation = 45, compare with it's upper-right and lower-left pixel
                if (magnitude[i, j]<=magnitude[i-1, j+1]) or (magnitude[i, j]<=magnitude[i+1, j-1]): # if these pixel's magnitude are all bigger than magnitude[i,j]
                    magnitude[i, j]=0 # set magnitude[i, j]=0
            elif orientation[i, j]==90: # if the pixel orientation = 90, compare with it's upper and lower pixel
                if (magnitude[i, j]<=magnitude[i+1, j]) or (magnitude[i, j]<=magnitude[i-1, j]): # if these pixel's magnitude are all bigger than magnitude[i,j]
                    magnitude[i, j]=0 # set magnitude[i, j]=0
            else: # if the pixel orientation = 135, compare with it's lower-right and upper-left pixel
                if (magnitude[i, j]<=magnitude[i+1, j+1]) or (magnitude[i, j]<=magnitude[i-1, j-1]): # if these pixel's magnitude are all bigger than magnitude[i,j]
                    magnitude[i, j]=0 # set magnitude[i, j]=0

    return magnitude

# ------ Function to do Edge Linking-Edge tracking by hysteresis  ------- #
def linking(i, j, M_above_high, M_above_low): # if pixel is an edge
    for m in range(-1, 2): 		# count the pixel around [i, j]
        for n in range(-1, 2):	# count the pixel around [i, j]
            if M_above_high[i+m, j+n]==0 and M_above_low[i+m, j+n]!=0: # if the pixel around [i, j]'s value is between upper and lower bound
                M_above_high[i+m, j+n]=1 # set that pixel to be edge
                linking(i+m, j+n, M_above_high, M_above_low) # do recursively to find next edge pixel

# ------------------- Function to do Hysteresis Thresholding ------------------- #
def Hysteresis_Thresholding(magnitude):
    m = np.max(magnitude) # find the largest pixel to be the parameter of the threshold

    ### upper:lower ratio between 2:1
    max_VAL = 0.2*m  # set upper bound
    min_VAL = 0.1*m  # set lower bound

    M_above_high=np.zeros((x,y), dtype='f') # initial the table with pixel value above upper bound are sure to be edges
    M_above_low=np.zeros((x,y), dtype='f')  # initial the table with pixel value above lower bound,
    									    # the pixel thich below the lower bound are sure to be non-edges

    # fill the pixel value in "M_above_high" and "M_above_low"
    for i in range(0, x):							 # count image pixel from 0 to x
        for j in range(0, y):						 # count image pixel from 0 to y
            if magnitude[i,j]>=max_VAL: 			 # if pixel magnitude value > upper bound
                M_above_high[i,j] = magnitude[i,j]	 # store to M_above_high
            if magnitude[i,j]>=min_VAL:				 # if pixel magnitude value > lower bound
                M_above_low[i,j] = magnitude[i,j]	 # store to M_above_low

    M_above_low = M_above_low - M_above_high # calculte the magnitude value which are less than uper bound and greater than lower bound
    										 # These are classified edges or non-edges based on their connectivity

    for i in range(1, x-1): 		# count pixel in M_above_high
        for j in range(1, y-1): 	# count pixel in M_above_high
            if M_above_high[i,j]: 	# if the pixel's value is greater than upper bound
                M_above_high[i,j]=1 # set [i,j] is an edge = 1
                linking(i, j, M_above_high, M_above_low) # call finction to find next edge pixel around [i, j]

    return M_above_high

# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # read one input from terminal
    # (1) command line >> python Canny_Edge_Detection.py  -i input_image.png

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(ap.parse_args())

    # Read image and convert to grayscale
    image = cv2.imread(args["image"])
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image to grayscal
    x, y = grayimg.shape # get image size x*y, a image with x rows and y columns

    ## Process Functions
    # Step 1. reduce noise
    Gimg = Image_Filtering(image)
    Gnonimg = Image_Padding(Gimg) # image padding for further image filtering

    # Step 2. Find Intensity Gradient and Orientation
    magnitude, orientation = Intensity_and_Orientation(Gnonimg)

    # Step 3. Non-maximum Suppression
    magnitude = Suppression(magnitude, orientation)

    # Step 4. Hysteresis Thresholding
    result = Hysteresis_Thresholding(magnitude)

    # show result image
    pl.subplot(121)				# image position
    pl.imshow(grayimg)			# show image "grayimg"
    pl.title('gray image')		# graph title "gray image"
    pl.set_cmap('gray')			# show in gray scale

    pl.subplot(122)				# image position
    pl.imshow(result)		    # show image "M_above_high"
    pl.title('edge image')		# graph title "edge image"
    pl.set_cmap('gray')			# show in gray scale

    pl.show()	                # output image
