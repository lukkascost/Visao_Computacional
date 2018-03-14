import cv2
import numpy as np

img = cv2.imread("img/lena.png", 0)
img_threshold = np.copy(img)

threshold_1 = 200
threshold_2 = 100

for i in range(img_threshold.shape[0]):
    for j in range(img_threshold.shape[1]):
        if threshold_2 <= img_threshold[i, j] < threshold_1:
            img_threshold[i,j] = 128
        elif img_threshold[i, j] > threshold_1:
            img_threshold[i,j] = 255
        else:
            img_threshold[i,j] = 0
cv2.imwrite("img/TRABALHO_10/LENA_MULTITHRESHOLD.png", img_threshold)
