#Import the required libraries for the project
#Make sure that all the following liberaries are installed

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 
import math
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
%matplotlib inline

--------------------------------------------------------------------------------------
# Camera calibration using chessboard corners dataset

wrldp = np.zeros((6*9,3), np.float32)
wrldp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

wrldpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibration/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    if found == True:
        wrldpoints.append(wrldp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, (9,6), corners, found)

        
def calibrate_camera(img, wrldpoints, imgpoints):
    
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(wrldpoints, imgpoints, img_size,None,None)
    
    return cv2.calibrateCamera(wrldpoints, imgpoints, img_size,None,None)

def cal_undistort(img):

    undis = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undis

img = mpimg.imread('calibration2/calibration2.jpg')
et, mtx, dist, rvecs, tvecs = calibrate_camera(img, wrldpoints, imgpoints)

--------------------------------------------------------------------------------------
#image processing pipeline

def abs_binary_sobl(image):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sob = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0,ksize=3))
    scaled = np.uint8(255*abs_sob/np.max(abs_sob))
    binary = np.zeros_like(scaled)
    binary [(scaled >= 15) & (scaled <= 200)] = 1
    
    return binary









