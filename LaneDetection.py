import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils
import time

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
    points = utils.val_trackbars()
    height, width, channels = 240, 480, 3  # frame.shape

    # create frames
    threshed_frame = utils.thresholding(frame)
    warped_and_threshed_frame = utils.warp(threshed_frame, points, width, height)
    frame_with_points = utils.draw_points(frame.copy(), points)
    # create mid histogram point
    base_point, hist_image_quarter = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.5,
                                                 display=True, region_percentage=4)
    curve_avg_point, hist_image_all = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.9,
                                                 display=True, display_base_point=False)
    curve_raw = curve_avg_point - base_point  # curve value

    # curve calculation
    curve_list.append(curve_raw)
    if len(curve_list) > curve_list_avg_len:
        curve_list.pop(0)
    curve = sum(curve_list) // len(curve_list)
    hist_image_quarter = utils.display_curve(hist_image_quarter, curve, fps)

    #display car
    car_image = cv2.imread('car2.png')
    car_y = car_image.shape[0]
    car_x = car_image.shape[1]
    hist_image_all[240 - car_y:240 - car_y + 113, base_point - int(car_x / 2):base_point + 135 - int(car_x / 2)] = car_image

    # visualize frames

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
