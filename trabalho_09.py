import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/lena.jpg", 0)
img_threshold = np.copy(img)

threshold = 125

img_threshold[img_threshold > threshold] = 255
img_threshold[img_threshold <= threshold] = 0

cv2.imshow("normal", img)
cv2.imshow("limiar", img_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
