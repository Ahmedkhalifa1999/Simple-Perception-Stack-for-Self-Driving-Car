import cv2 as cv
import sys
from modules.lane_detection import detect_lanes
from modules.preprocessing import preprocess
from modules.region_of_interest import perspective_transform
from modules.postprocessing import postprocess
from modules.line_finding import *
from modules.lane_metrics import *

def process_frame(image):

    equalized_image = cv.cvtColor(preprocess(image), cv.COLOR_BGR2HSV)
    warped_image, Minv = perspective_transform(equalized_image)
    binary_warped = detect_lanes(warped_image)
    
    left_x, right_x, ploty, _ = fit_polynomial(binary_warped)
    
    left_curvature, right_curvature = measure_curvature(binary_warped)
    center = get_center_distance(binary_warped)
    
    drawn_img = draw_lane(image,binary_warped, left_x, right_x, ploty,Minv)
    drawn_img_with_values = draw_values(drawn_img,left_curvature,right_curvature, center)
    
    return drawn_img_with_values


if __name__ == "__main__":
    #video_reader = cv.VideoCapture(sys.argv[1])
    #video_writer = cv.VideoWriter(sys.argv[2], *'DIVX', (video_reader.get(cv.CAP_PROP_FRAME_WIDTH), video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)), video_reader.get(cv.CAP_PROP_FPS))
    video_reader = cv.VideoCapture("Data/Harder Challenge video.mp4")
    frame_width = video_reader.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_size = tuple([frame_width, frame_height])
    video_writer = cv.VideoWriter('Output/Harder Challenge video.mp4', cv.VideoWriter_fourcc(*'DIVX'), video_reader.get(cv.CAP_PROP_FPS), (1280, 720))
    print("Created Video Reader and Writer")
    while True:
        ret, frame = video_reader.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        processed_frame = process_frame(frame)
        video_writer.write(processed_frame)
