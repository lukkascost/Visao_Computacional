import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/01.png", 0)
histogram = np.zeros(256)

for i in range(256):
    histogram[i] = len(img[img == i])

plt.plot(histogram)
plt.show()
