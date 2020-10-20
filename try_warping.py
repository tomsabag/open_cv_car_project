import numpy as np
import cv2

cards = cv2.imread('cards.jpg')
width = 250
height = 350
pts = np.float32([[148, 127], [225, 168], [216, 15], [287, 59]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

for i in range(4):
    cv2.circle(cards, (pts[i][0], pts[i][1]), 3, (0, 0, 255), -1)

matrix = cv2.getPerspectiveTransform(pts, pts2)
new_img = cv2.warpPerspective(cards, matrix, (width, height))
cv2.imshow('cards', new_img)
cv2.waitKey(0)
