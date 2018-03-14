import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/lena.png", 0)
img_threshold = np.copy(img)

threshold = 125

img_threshold[img_threshold > threshold] = 255
img_threshold[img_threshold <= threshold] = 0

cv2.imwrite("img/TRABALHO_09/LENA_THRESHOLD.png", img_threshold)
