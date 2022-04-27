import numpy as np
import cv2 as cv

def perspective_transform(img):
    
    img_size = (img.shape[1], img.shape[0])
    x_dim, y_dim = img.shape[1], img.shape[0]

    middle_x = x_dim//2

    top_y = 2*y_dim//3

    top_margin = 95

    bottom_margin = 470
    
    points = [
        (middle_x-top_margin, top_y),
        (middle_x+top_margin, top_y),
        (middle_x+bottom_margin, y_dim),
        (middle_x-bottom_margin, y_dim)
    ]
    src = np.float32(points)
    
    dst = np.float32([
        (middle_x-bottom_margin, 0),
        (middle_x+bottom_margin, 0),
        (middle_x+bottom_margin, y_dim),
        (middle_x-bottom_margin, y_dim)
    ])

    M = cv.getPerspectiveTransform(src, dst)
    inv_M = cv.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M, inv_M