import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/lena.jpg", 0)
histogram = np.zeros(256)

for i in range(256):
    histogram[i] = len(img[img == i])
plt.plot(histogram)
minim = 0
maxim = 0
cut = 10
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
img_realce = np.multiply(img_realce, img).astype(np.uint8)

histogram_equalized = np.zeros(256)

for i in range(256):
    histogram_equalized[i] = len(img_realce[img_realce == i])
plt.plot(histogram_equalized)

plt.show()

cv2.imshow("normal", img)
cv2.imshow("realce", img_realce)

cv2.waitKey(0)
cv2.destroyAllWindows()