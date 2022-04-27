import cv2 as cv
import sys
from modules.thresholding import get_optimal_threshold
from modules.perspective_transform import perspective_transform
from modules.line_finding import fit_polynomial
from modules.lane_metrics import measure_curvature, get_center_distance, draw_lane, draw_values
from modules.lane_class import blind_search_class, draw_lane_class

def detect_lane_pipeline(image):

    threshed = get_optimal_threshold(image)
    warped, M, Minv = perspective_transform(threshed)
    
    left_x, right_x, ploty, out_image  = fit_polynomial(warped)
    
    left_curvature, right_curvature = measure_curvature(warped)
    center = get_center_distance(warped)
    
    drawn_img = draw_lane(image,warped, left_x, right_x, ploty,Minv)
    drawn_img_with_values = draw_values(drawn_img,left_curvature,right_curvature, center)
    
    return drawn_img_with_values, left_curvature, right_curvature, center

def detect_lane_pipe_class(image):
    
    threshed = get_optimal_threshold(image)
    
    warped, M, inv_perspective_M = perspective_transform(threshed)

    lane = blind_search_class(warped)
    
    drawn_img = draw_lane_class(image,lane)
    
    return drawn_img


if __name__ == "__main__":
    #video_reader = cv.VideoCapture(sys.argv[1])
    #video_writer = cv.VideoWriter(sys.argv[2], *'DIVX', (video_reader.get(cv.CAP_PROP_FRAME_WIDTH), video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)), video_reader.get(cv.CAP_PROP_FPS))
    video_reader = cv.VideoCapture("Data/Main Video.mp4")
    frame_width = video_reader.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_size = tuple([frame_width, frame_height])
    video_writer = cv.VideoWriter('Output/Main Video.mp4', cv.VideoWriter_fourcc(*'DIVX'), video_reader.get(cv.CAP_PROP_FPS), (1280, 720))
    print("Created Video Reader and Writer")
    while True:
        ret, frame = video_reader.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        processed_frame = detect_lane_pipe_class(frame)
        video_writer.write(processed_frame)
