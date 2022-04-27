import numpy as np
import cv2 as cv

def perspective_transform(image):
    src = np.float32([[375, 480],
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
    return warped, Minv