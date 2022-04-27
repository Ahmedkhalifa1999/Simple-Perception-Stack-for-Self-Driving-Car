from math import fabs
from pickletools import uint8
import numpy as np
import cv2 as cv
from modules.line_finding import blind_search, fit_polynomial

def measure_curvature(binary_warped):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # data detection using blind search
    leftx, lefty, rightx, righty, out_img = blind_search(binary_warped)
    #fit poly
    if (len(leftx) == 0)| (len(lefty) == 0) |(len(rightx) == 0)| (len(righty) == 0) :
        left_fit_cr =[0,0,0]
        right_fit_cr =[0,0,0]
    else:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = binary_warped.shape[0]
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def get_center_distance(binary_warped):
    
    left_fitx, right_fitx, ploty, out_image = fit_polynomial(binary_warped)
    lane_width = right_fitx[-1] - left_fitx[-1]
    lane_mid = lane_width/2 + left_fitx[-1]
    lane_width_m = 3.7
    center_distance = lane_width_m*((lane_mid- binary_warped.shape[1]//2)/lane_width)
    
    return center_distance

def draw_lane(original_img,binary_warped, left_fitx,right_fitx,ploty, inv_perspective_M):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, inv_perspective_M, (original_img.shape[1], original_img.shape[0])) 
    # Combine the result with the original image
    result = cv.addWeighted(original_img, 1, newwarp, 0.3, 0)

    return result

def draw_values(img,left_curvature,right_curvature, center_distance):
    
        offset_y = 100
        offset_x = 100
        
        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center offset and Alignment")
        txt_values = template.format("{:.4f}m".format(left_curvature),
                                     "{:.4f}m".format(right_curvature),
                                     "{:.4f}m Right".format(center_distance))
        if center_distance < 0.0:
            txt_values = template.format("{:.4f}m".format(left_curvature),
                                         "{:.4f}m".format(right_curvature),
                                         "{:.4f}m Left".format(fabs(center_distance)))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, txt_header, (offset_x, offset_y), font, 1, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(img, txt_values, (offset_x, offset_y + 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        return img