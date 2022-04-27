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
    
    return drawn_img_with_values, warped_image, binary_warped


if __name__ == "__main__":
    #video_reader = cv.VideoCapture(sys.argv[1])
    #video_writer = cv.VideoWriter(sys.argv[2], *'DIVX', (video_reader.get(cv.CAP_PROP_FRAME_WIDTH), video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)), video_reader.get(cv.CAP_PROP_FPS))
    video_reader = cv.VideoCapture("Data/Harder Challenge Video.mp4")
    frame_width = video_reader.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_size = tuple([frame_width, frame_height])
    video_writer = cv.VideoWriter('Output/Harder Challenge Video.mp4', cv.VideoWriter_fourcc(*'DIVX'), video_reader.get(cv.CAP_PROP_FPS), (1280, 720))
    print("Created Video Reader and Writer")
    while True:
        ret, frame = video_reader.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        processed_frame, warped_frame, binary_warped_frame = process_frame(frame)

        debug_frame_upper = np.concatenate((cv.resize(frame, (640, 360)), cv.resize(processed_frame, (640, 360))), axis = 1)
        binary_warped_frame[binary_warped_frame] = 255
        debug_frame_lower = np.concatenate((cv.cvtColor(cv.resize(warped_frame, (640, 360)), cv.COLOR_HSV2BGR), 
                                            cv.cvtColor(cv.resize(binary_warped_frame, (640, 360)), cv.COLOR_GRAY2BGR)), 
                                            axis = 1)
        debug_frame = np.concatenate((debug_frame_upper, debug_frame_lower), axis = 0)

        video_writer.write(debug_frame)
