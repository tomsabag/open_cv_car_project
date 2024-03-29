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


def lane_curve(frame, width, height):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    warping_points = utils.val_trackbars()

    # create frames
    threshed_frame = utils.thresholding(frame)
    warped_and_threshed_frame = utils.warp(threshed_frame, warping_points, width, height)
    frame_with_points = utils.draw_points(frame.copy(), warping_points)
    # create base point and curvature
    base_point, hist_image_quarter = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.5,
                                                 display=True, region_percentage=4)
    curve_avg_point, hist_image_all = utils.get_histogram(warped_and_threshed_frame, min_thresh_percent=0.9,
                                                 display=True, display_base_point=False)
    curve_raw = curve_avg_point - base_point  # curve value

    curve_list.append(curve_raw)
    if len(curve_list) > curve_list_avg_len:
        curve_list.pop(0)
    curve = sum(curve_list) // len(curve_list)
    hist_image_quarter = utils.display_curve(hist_image_quarter, curve, fps)

    # draw angle
    hist_image_all = utils.draw_angle(hist_image_all, curve)

    # display car
    hist_image_all = utils.draw_car(hist_image_all, height, base_point)

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
    warped_and_threshed_frame = lane_curve(frame, width, height)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
