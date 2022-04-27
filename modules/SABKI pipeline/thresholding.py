import numpy as np
import cv2 as cv

def abs_sobelx_thresh(image):
    
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # Apply x  gradient with the OpenCV Sobel() function & take the absolute value
    abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0,ksize=3))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= 15) & (scaled_sobel <= 200)] = 1
    
    return grad_binary

def dir_threshold(image):

    # 1) Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=19)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=19)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction_mag = np.arctan2(abs_sobely , abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction_mag)
    dir_binary[(direction_mag >= np.pi/6) & (direction_mag <= np.pi/2)] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary

def treshold_S_channel(image):
    
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold S_channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1

    return s_binary

def Red_Green_tresh(image):
    
    R = image[:,:,0]
    G = image[:,:,1]
    r_g_combined = np.zeros_like(R)
    r_g_combined = (R > 155) & (G > 155)
    
    return r_g_combined

def treshold_L_channel(image):
    
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 120) & (l_channel <= 255)] = 1

    return l_binary

def get_optimal_threshold(image):

    sobel_x_binary = abs_sobelx_thresh(image)
    dir_binary = dir_threshold(image)
    combined_grad_dir = ((sobel_x_binary == 1) & (dir_binary == 1))
    
    S_binary = treshold_S_channel(image)
    L_binary = treshold_L_channel(image)
    R_G_binary = Red_Green_tresh(image)
    
    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    R = image[:,:,0]
    optimal_treshold = np.zeros_like(image)
    optimal_treshold = (((R_G_binary== 1) & (L_binary== 1)) & ((S_binary== 1) | (combined_grad_dir== 1)))
    optimal_treshold = np.array(optimal_treshold, dtype=np.uint8)
    
    return optimal_treshold

