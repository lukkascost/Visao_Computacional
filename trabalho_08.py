import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/lena.png", 0)
histogram = np.zeros(256)

for i in range(256):
    histogram[i] = len(img[img == i])
#plt.bar([x for x in range(256)], histogram)
minim = 0
maxim = 0
cut = 20
for i, j in enumerate(histogram):
    if j < cut:
        pass
    else:
        minim = i
        break
for i in range(histogram.shape[0] - 1, 0, -1):
    if histogram[i] < cut:
        pass
    else:
        maxim = i
        break

img_realce = img - minim
img_realce = img_realce.astype(float) / (maxim - minim)
img_realce = np.multiply(img_realce, 255).astype(np.uint8)

histogram_equalized = np.zeros(256)

for i in range(256):
    histogram_equalized[i] = len(img_realce[img_realce == i])
plt.bar([x for x in range(256)], histogram_equalized)

plt.savefig("img/TRABALHO_08/LENA_REALCE_HISTOGRAM.png")
cv2.imwrite("img/TRABALHO_08/LENA_REALCE.png", img_realce)
