# Shivpure, Sameer
# 1001-417-543
# 2017-04-17
# Final_code_02

# import required python packages

import numpy as np
import cv2 as cv
from sklearn.externals import joblib
import os

# path to the configuration files and input video
hog_config = 'Shivpure_hogxml_01.xml'
svm_model = 'Shivpure_svm_01.data'
video_src = input('Please enter the input video file path..\n')

if (os.path.exists(video_src)):
    print("Please wait...")
else:
    print("File dose not exist! Please check the path.")
    exit()

# load the hog descriptor, svm and video capture object
hog = cv.HOGDescriptor(hog_config)
SVM = joblib.load(svm_model)
vid_handler = cv.VideoCapture(video_src)
Bckg_subs = cv.createBackgroundSubtractorMOG2(detectShadows=False)
window_stride = 10
frame_no = 0
track_points = []


# function to detect the empty parking space
def detectEmptyParkingSpaces(image):

    imr = image
    pstride = 10
    emt_park_sp = []
    for i in range(2):
        sx = image.shape[1] / imr.shape[1]
        sy = image.shape[0] / imr.shape[0]
        x = np.arange(0, imr.shape[1], pstride)
        y = np.arange(0, imr.shape[0], pstride)

        for block_y in y:
            for block_x in x:
                if block_x + 24 < imr.shape[1] and block_y + 24 < imr.shape[0]:
                    block = imr[block_y: block_y + 24, block_x: block_x + 24, :]
                    edges = cv.Canny(block, 100, 150)
                    stripes = cv.HoughLines(edges, 1, np.pi / 180, 24)
                    if stripes is not None and len(stripes) == 2:
                        _, contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            rect = cv.boundingRect(cnt)
                            if rect[2] > 19 or rect[3] > 19:
                                emt_park_sp.append([int((block_x + 15) * sx), int((block_y) * sy), int((block_x + 40) * sx), int((block_y + 24) * sy)])
                                break

        imr = cv.resize(imr, (int(imr.shape[1] / 1.25), int(imr.shape[0] / 1.25)))

    return emt_park_sp

# read individual frame of the video in a loop
while vid_handler.isOpened():
    rec, frame = vid_handler.read()
    if (type(frame) == type(None)):
        break

    frame_no += 1

    # resize the frame to lower dimension for faster processing
    resize_dimen = (int(frame.shape[1] * 0.55), int(frame.shape[0] * 0.6))
    if frame.shape[1] > 1000 and frame.shape[0] > 1000:
        frame_resized_orig = cv.resize(frame, resize_dimen)
    else:
        frame_resized_orig = frame
    # Apply gaussian filter to reduce the noise and convert the frame image to grayscale
    frame_resized = cv.GaussianBlur(frame_resized_orig, (3, 3), 1)
    # Background subtraction to obtain foreground moving object
    img = Bckg_subs.apply(frame_resized)

    # For the first frame of video, use the frame as background image and continue without detections
    if frame_no == 1:
        mask = np.zeros_like(frame_resized)
        continue

    # Detect the positions of vehicles in the new frame after a interval of every 10 frames
    imt = img
    car_detections = []
    if (frame_no - 2) % 10 == 0:
        for scale_index in range(2):
            sx = img.shape[1] / imt.shape[1]
            sy = img.shape[0] / imt.shape[0]
            x = np.arange(0, imt.shape[1], window_stride)
            y = np.arange(0, imt.shape[0], window_stride)
            # extract hog feature for each window of size 24 x 24
            for block_y in y:
                for block_x in x:
                    if block_x + 24 < imt.shape[1] and block_y + 24 < imt.shape[0]:
                        block = imt[block_y: block_y + 24, block_x: block_x + 24]
                        if np.max(block) == 0:
                           continue
                        histogram = hog.compute(block, winStride=(8, 8), padding=(2, 2))
                        # Detect a vehicle in the window using SVM classifier on the histogram of gradients
                        pred = SVM.predict(np.transpose(histogram))

                        if pred == 1:
                            car_detections.append([int((block_x) * sx), int((block_y) * sy)])
            imt = cv.resize(imt, (int(imt.shape[1] / 1.25), int(imt.shape[0] / 1.25)))
        parkingspace_detections = detectEmptyParkingSpaces(frame_resized_orig)
        # Add the frame and detected points as points to be tracked in optical flow
        prev_frame_gray = img
        if len(car_detections) > 0:
            car_detections = np.array(car_detections)
            track_points = (car_detections + 12).reshape(-1, 1, 2)
            track_points = np.float32(track_points)
        continue

    # Optical flow motion detection to follow the path of moving vehicles
    curr_points, status, error = cv.calcOpticalFlowPyrLK(prev_frame_gray, img, track_points, None, winSize=(15, 15), maxLevel=1, minEigThreshold=0.75)

    for i, (curr, old) in enumerate(zip(curr_points[status == 1], track_points[status == 1])):
        a, b = curr.ravel()
        c, d = old.ravel()
        cv.line(mask, (a, b), (c, d), (0, 0, 178), 4)
    frame_resized_orig = cv.add(frame_resized_orig, mask)

    for (x, y, w, h) in parkingspace_detections:
         cv.rectangle(frame_resized_orig, (x, y), (w, h), (20, 140, 0), 2)

    cv.putText(frame_resized_orig, 'Car tracking', (int(frame_resized_orig.shape[1]*0.45), int(frame_resized_orig.shape[0]*0.09)), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 178), 2, cv.LINE_AA)
    cv.putText(frame_resized_orig, 'Parking space', (int(frame_resized_orig.shape[1]*0.45), int(frame_resized_orig.shape[0]*0.12)), cv.FONT_HERSHEY_PLAIN, 1, (20, 140, 0), 2, cv.LINE_AA)
    cv.imshow('Image', frame_resized_orig)
    cv.waitKey(3000)
    key = cv.waitKey(30) & 0xff
    if key == 27:
        break

cv.destroyAllWindows()