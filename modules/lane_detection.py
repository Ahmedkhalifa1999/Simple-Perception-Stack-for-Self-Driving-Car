import numpy as np
import cv2 as cv

def detect_lanes(image):
    
    h, s, v = cv.split(image)
    _, white_s_thresholded = cv.threshold(s, 5, 255, cv.THRESH_BINARY_INV)
    _, white_v_thresholded = cv.threshold(v, 200, 255, cv.THRESH_BINARY)
    yellow_h_thresholded = cv.bitwise_and(cv.threshold(h, 10, 255, cv.THRESH_BINARY)[1], cv.threshold(h, 50, 255, cv.THRESH_BINARY_INV)[1])
    _, yellow_s_thresholded = cv.threshold(s, 100, 255, cv.THRESH_BINARY)
    _, yellow_v_thresholded = cv.threshold(s, 100, 255, cv.THRESH_BINARY)
    white_mask = cv.bitwise_and(white_s_thresholded, white_v_thresholded)
    yellow_mask = cv.bitwise_and(cv.bitwise_and(yellow_h_thresholded, yellow_s_thresholded), yellow_v_thresholded)
    lane_mask = cv.bitwise_or(white_mask, yellow_mask)
    # lane_mask[0:100, :] = 0
    # lane_mask[400:500, :] = 0
    return lane_mask
        