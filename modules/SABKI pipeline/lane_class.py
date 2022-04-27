import math
import numpy as np
import cv2 as cv
from modules.lane_metrics import get_center_distance, measure_curvature
from modules.perspective_transform import perspective_transform

class Lane():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.binary_warped = None
        self.left_curvature = None
        self.right_curvature = None
        self.center_distance = None

def blind_search_class(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 200

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        left_inds = (
            (nonzeroy >= win_y_low)&
            (nonzeroy < win_y_high)&
            (nonzerox >= win_xleft_low)&
            (nonzerox < win_xleft_high)
        ).nonzero()[0]
        
        right_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low)&
            (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[left_inds]))
        if len(right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    left_X = nonzerox[left_lane_inds]
    left_Y = nonzeroy[left_lane_inds] 
    right_X = nonzerox[right_lane_inds]
    right_Y = nonzeroy[right_lane_inds]
    
    #left_fit = np.polyfit(left_Y, left_X, 2)
    #right_fit = np.polyfit(right_Y, right_X, 2)
    
    # Fit a second order polynomial to each using `np.polyfit`
    if (len(left_X) == 0)| (len(left_Y) == 0) |(len(right_X) == 0)| (len(right_Y) == 0) :
        left_fit =[0,0,0]
        right_fit =[0,0,0]
    else:
        left_fit = np.polyfit(left_Y, left_X, 2)
        right_fit = np.polyfit(right_Y, right_X, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    #left_curvature, right_curvature = measure_curvature(binary_warped)
    #center = get_center_distance(binary_warped)
    left_curvature, right_curvature = measure_curvature(binary_warped)
    center = get_center_distance(binary_warped)
    
    lane = Lane()
    
    lane.left_fit = left_fit
    lane.right_fit = right_fit
    lane.ploty = ploty
    lane.left_fitx = left_fitx
    lane.right_fitx = right_fitx
    lane.binary_warped = binary_warped
    lane.left_curvature = left_curvature
    lane.right_curvature = right_curvature
    lane.center_distance = center
        
    return lane

def draw_lane_class(original_img,lane):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(lane.binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lane.left_fitx, lane.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lane.right_fitx, lane.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    _, _, inv_perspective_M = perspective_transform(color_warp)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, inv_perspective_M, (lane.binary_warped.shape[1], lane.binary_warped.shape[0])) 
    # Combine the result with the original image
    lane_drawn = cv.addWeighted(original_img, 1, newwarp, 0.3, 0)
    lane_drawn = draw_values_class(lane_drawn,lane)

    return lane_drawn

def draw_values_class(img,lane):
    
        offset_y = 100
        offset_x = 100

        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center offset and Alignment")
        # print(txt_header)
        txt_values = template.format("{:.4f}m".format(lane.left_curvature),
                                     "{:.4f}m".format(lane.right_curvature),
                                     "{:.4f}m Right".format(lane.center_distance))
        if lane.center_distance < 0.0:
            txt_values = template.format("{:.4f}m".format(lane.left_curvature),
                                         "{:.4f}m".format(lane.right_curvature),
                                         "{:.4f}m Left".format(math.fabs(lane.center_distance)))

        # print(txt_values)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, txt_header, (offset_x, offset_y), font, 1, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(img, txt_values, (offset_x, offset_y + 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        return img