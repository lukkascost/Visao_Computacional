import cv2
import numpy as np

orig = cv2.imread("img/fill2.png", 0)

img = np.copy(orig)
# img = cv2.Canny(img, 50, 150, apertureSize=3)
cv2.imwrite("canny.png", img)

for it in range(240):
    menor = np.min(img)
    img[img == menor] += 1
    cv2.imwrite("it/result{:04d}.png".format(it), img)
