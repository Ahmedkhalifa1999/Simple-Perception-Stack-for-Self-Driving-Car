import cv2 as cv
import sys
import os
from modules.lane_detection import detect_lanes
from modules.preprocessing import preprocess
from modules.region_of_interest import perspective_transform
from modules.line_finding import *
from modules.lane_metrics import *

yolo = cv.dnn.readNetFromDarknet(os.path.join("YOLO", "yolov3.cfg"), os.path.join("YOLO", "yolov3.weights"))

def process_frame(image):

    equalized_image = cv.cvtColor(preprocess(image), cv.COLOR_BGR2HSV)
    warped_image, Minv = perspective_transform(equalized_image)
    binary_warped = detect_lanes(warped_image)
    
    left_x, right_x, ploty, _ = fit_polynomial(binary_warped)
    
    left_curvature, right_curvature = measure_curvature(binary_warped)
    center = get_center_distance(binary_warped)
    
    drawn_img = draw_lane(image,binary_warped, left_x, right_x, ploty,Minv)
    drawn_img_with_values = draw_values(drawn_img,left_curvature,right_curvature, center)

    detected = drawn_img_with_values.copy()

    names = yolo.getLayerNames()
    (H, W) = image.shape[:2]
    output_layers_names = [names[i - 1] for i in yolo.getUnconnectedOutLayers()]
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), crop=False, swapRB=False)
    yolo.setInput(blob)
    layers_output = yolo.forward(output_layers_names)

    boxes = []
    confidences = []
    classIDs = []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.85:
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype(int)
                x = int(bx - bw/2)
                y = int(by - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                classIDs.append(classID)
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)
    labels = open(os.path.join("YOLO", "coco.names")).read().strip().split('\n')
    for i in idxs:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv.rectangle(detected, (x,y), (x + w, y + h), (255, 255, 0), 2)
        cv.putText(detected, "{}: {:.2f}".format(labels[classIDs[i]], confidences[i]), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return detected, drawn_img_with_values, warped_image, binary_warped

def main(input_path, output_path, debugging):
    #video_reader = cv.VideoCapture(sys.argv[1])
    #video_writer = cv.VideoWriter(sys.argv[2], *'DIVX', (video_reader.get(cv.CAP_PROP_FRAME_WIDTH), video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)), video_reader.get(cv.CAP_PROP_FPS))
    video_reader = cv.VideoCapture(input_path)
    frame_width = video_reader.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_size = (int(frame_width), int(frame_height))
    video_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), video_reader.get(cv.CAP_PROP_FPS), frame_size)
    print("Created Video Reader and Writer")
    while True:
        ret, frame = video_reader.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        detected_image, processed_frame, warped_frame, binary_warped_frame = process_frame(frame)
        if debugging == 1:
            debug_frame_upper = np.concatenate((cv.resize(frame, (640, 360)), cv.resize(processed_frame, (640, 360))), axis = 1)
            binary_warped_frame[binary_warped_frame] = 255
            debug_frame_lower = np.concatenate((cv.cvtColor(cv.resize(warped_frame, (640, 360)), cv.COLOR_HSV2BGR), 
                                                cv.cvtColor(cv.resize(binary_warped_frame, (640, 360)), cv.COLOR_GRAY2BGR)), 
                                                axis = 1)
            debug_frame = np.concatenate((debug_frame_upper, debug_frame_lower), axis = 0)
            video_writer.write(debug_frame)
        else:
            video_writer.write(detected_image)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        debugging = sys.argv[3]
    else:
        debugging = 0
    main(input_path, output_path, debugging)