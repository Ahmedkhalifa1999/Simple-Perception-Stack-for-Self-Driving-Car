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
#image processing pipeline

def abs_binary_sobl(image):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sob = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0,ksize=3))
    scaled = np.uint8(255*abs_sob/np.max(abs_sob))
    binary = np.zeros_like(scaled)
    binary [(scaled >= 15) & (scaled <= 200)] = 1
    
    return binary

def dir_sobl(image):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    soblx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=19)
    sobly = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=19)
    # 3) Take the absolute value of the x and y gradients
    abs_soblx = np.absolute(soblx)
    abs_sobly = np.absolute(sobly)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_mag = np.arctan2(abs_sobly , abs_soblx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_mag)
    dir_binary[(dir_mag >= np.pi/6) & (dir_mag <= np.pi/2)] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary

def treshold_S_channel(image):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold S_channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1

    return s_binary

def Red_Green_tresh(image):
    
    R = image[:,:,0]
    G = image[:,:,1]
    R_G_combined = np.zeros_like(R)
    r_g_combined = (R > 155) & (G > 155)
    
    return r_g_combined

def treshold_L_channel(image):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 120) & (l_channel <= 255)] = 1

    return l_binary

def get_optimal_treshold(image):

    sobel_x_binary = abs_sobelx_thresh(image)
    dir_binary = dir_threshold(image)
    combined_grad_dir = ((sobel_x_binary == 1) & (dir_binary == 1))
    
    S_binary = treshold_S_channel(image)
    L_binary = treshold_L_channel(image)
    R_G_binary = Red_Green_tresh(image)
    
    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    R = img[:,:,0]
    optimal_treshold = np.zeros_like(image)
    optimal_treshold = (((R_G_binary== 1) & (L_binary== 1)) & ((S_binary== 1) | (combined_grad_dir== 1)))
    optimal_treshold = np.array(optimal_treshold, dtype=np.uint8)
    
    return optimal_treshold


