import cv2
import numpy as np

img = cv2.imread("img/lena.jpg", 0)
img_threshold = np.copy(img)

threshold_1 = 200
threshold_2 = 100


img_threshold[img_threshold < threshold_1] = 255
img_threshold[threshold_1 <= img_threshold < threshold_2] = 128
img_threshold[img_threshold >= threshold_2] = 0

cv2.imshow("normal", img)
cv2.imshow("multilimiar", img_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
