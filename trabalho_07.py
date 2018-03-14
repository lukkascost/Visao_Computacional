import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/lena.png", 0)
histogram = np.zeros(256)

for i in range(256):
    histogram[i] = len(img[img == i])

plt.bar([x for x in range(256)], histogram)
plt.savefig("img/TRABALHO_07/LENA_HISTOGRAM.png")
