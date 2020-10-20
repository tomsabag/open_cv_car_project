import cv2
import numpy as np

def nothing():
    pass
img = np.zeros((500, 300, 3))
cv2.namedWindow('play')

cv2.createTrackbar('B', 'play', 0, 255, nothing)
cv2.createTrackbar('G', 'play', 0, 255, nothing)
cv2.createTrackbar('R', 'play', 0, 255, nothing)

while True:
    cv2.imshow('play', img)

    b = cv2.getTrackbarPos('B', 'play')
    g = cv2.getTrackbarPos('G', 'play')
    r = cv2.getTrackbarPos('R', 'play')

    img[:] = [b, g, r]

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
