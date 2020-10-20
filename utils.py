import cv2
import numpy as np


def thresholding(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 0, 0])
    upper = np.array([255, 160, 255])
    mask_white = cv2.inRange(hsv_frame, lower, upper)
    return mask_white


def warp(frame, points, width, height, inv=False):
    pts1 = np.float32([points])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output_frame = cv2.warpPerspective(frame, matrix, (width, height))
    return output_frame


def nothing(x):
    pass


def initialize_Trackbars(initial_trackbar_vals,w=480, h=240):
    cv2.namedWindow('Warping Points Trackbars')
    cv2.resizeWindow('Warping Points Trackbars', 360, 240)
    cv2.createTrackbar('Width Top', 'Warping Points Trackbars', initial_trackbar_vals[0], w // 2, nothing)
    cv2.createTrackbar('Height Top', 'Warping Points Trackbars', initial_trackbar_vals[1], h, nothing)
    cv2.createTrackbar('Width Bottom', 'Warping Points Trackbars', initial_trackbar_vals[2], w // 2, nothing)
    cv2.createTrackbar('Height Bottom', 'Warping Points Trackbars', initial_trackbar_vals[3], h, nothing)


def val_trackbars(w=480, h=240):
    width_top = cv2.getTrackbarPos('Width Top', 'Warping Points Trackbars')
    height_top = cv2.getTrackbarPos('Height Top', 'Warping Points Trackbars')
    width_bottom = cv2.getTrackbarPos('Width Bottom', 'Warping Points Trackbars')
    height_bottom = cv2.getTrackbarPos('Height Bottom', 'Warping Points Trackbars')
    points = np.float32([(width_top, height_top), (w - width_top, height_top),
                         (width_bottom, height_bottom), (w - width_bottom, height_bottom)])
    return points


def draw_points(frame, points):
    for i in range(4):
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 15, (0, 255, 0), -1)
    return frame


def get_histogram(frame, min_thresh_percent=0.1, display=False, region_percentage=1):
    if region_percentage == 1:
        hist_vals = np.sum(frame, axis=0)
    else:
        hist_vals = np.sum(frame[frame.shape[0] // region_percentage:, :], axis=0)

    max_val = np.max(hist_vals)

    min_threshold = max_val * min_thresh_percent

    indices = np.where(hist_vals >= min_threshold)
    base_point = int(np.average(indices))

    if display:
        hist_img = np.zeros_like(frame)
        hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)

        for i, intensity, in enumerate(hist_vals):
            cv2.line(hist_img, (i, hist_img.shape[0]), (i, hist_img.shape[0] - intensity // 255 // region_percentage), (255, 0, 255), 5)
            cv2.circle(hist_img, (base_point, hist_img.shape[0]), 20, (0, 255, 255), -1)
        return base_point, hist_img

    return base_point


def display_curve(frame, curve, fps):
    height, width, channels = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, 'FPS:' + str(fps), (20, 50), font, 1, (255, 0, 0), 1)
    cv2.line(frame, (width // 10, int((height * 0.9))), (int(width * 0.9) - 5, int((height * 0.9))), (0, 0, 255), 3)

    for x in range(width // 10 + 2, int(width * 0.9), 25):
        cv2.line(frame, (x, int(height * 0.85)), (x, int(height * 0.95)), (0, 0, 255), 3)

    cv2.circle(frame, (width // 2, int(0.9 * height)), 6, (255, 255, 0), -1)

    cv2.arrowedLine(frame, (width // 2, int(0.9 * height)), (width // 2 + (curve * 2), int(0.9 * height)), (255, 255, 0), 5)
    cv2.putText(frame, str(curve), (width // 2 - 50, int(height * 0.9) - 15), font, 1, (255, 255, 0), 3)
    return frame
