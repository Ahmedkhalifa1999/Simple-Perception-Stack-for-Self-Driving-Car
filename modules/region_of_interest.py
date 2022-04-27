import numpy as np
import cv2 as cv

def perspective_transform(image):
    """ src = np.float32([[375, 480],
                    [905, 480],
                    [1811, 685],
                    [-531, 685]])
    dst = np.float32([[0, 0],
                    [500, 0],
                    [500, 600],
                    [0, 600]])
                    
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(image, M, (500, 600))
    return warped, Minv """
    img_size = (image.shape[1], image.shape[0])
    offset = 100
    x_dim, y_dim = image.shape[1], image.shape[0]

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
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(image, M, img_size)

    return warped, Minv