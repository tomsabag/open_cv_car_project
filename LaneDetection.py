import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import utils

delete_this_variable = [False]

cap = cv2.VideoCapture('lane_video.mp4')
# Adjust width and height according to rasberry pi camera
width = 480
height = 240
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_counter = 0

curve_list = []
curve_list_avg_len = 10

initial_points_trackbar_values = [102, 80, 20, 214]
utils.initialize_Trackbars(initial_points_trackbar_values)


def lane_curve(frame):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_copy = frame.copy()

    threshed_frame = utils.thresholding(frame)
    height, width, channels = 240, 480, 3  # frame.shape

    points = utils.val_trackbars()
    warped_frame = utils.warp(frame_copy, points, width, height)
    warped_and_threshed_frame = utils.warp(threshed_frame, points, width, height)
    warped_frame_with_points = utils.draw_points(warped_frame, points)
    frame_with_points = utils.draw_points(frame.copy(), points)
    middle_point, hist_image_quarter = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.5,
                                                 display=True, region_percentage=4)
    curve_avg_point, hist_image_all = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.9,
                                                 display=True)
    curve_raw = curve_avg_point - middle_point

    curve_list.append(curve_raw)
    if len(curve_list) > curve_list_avg_len:
        curve_list.pop(0)
    curve = sum(curve_list) // len(curve_list)
    hist_image_quarter = utils.display_curve(hist_image_quarter, curve_raw, fps)

    frames_stack1 = cv2.hconcat([hist_image_quarter, hist_image_all, frame_with_points])
    frames_stack2 = cv2.hconcat([threshed_frame, warped_and_threshed_frame])

    cv2.imshow('Histogram and Original video', frames_stack1)
    cv2.imshow('Thresholding and Warping', frames_stack2)

    return warped_and_threshed_frame


while True:
    # loop video when it finishes
    frame_counter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frame_counter:
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    ret, frame = cap.read()
    #time.sleep(fps)
    frame = cv2.resize(frame, (width, height))
    warped_and_threshed_frame = lane_curve(frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
