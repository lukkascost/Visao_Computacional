import cv2
import numpy as np

orig = cv2.imread("img/fill.png", 0)
print orig.shape

img = cv2.floodFill(orig, None, (400,20), 0)
cv2.imwrite("result.png", orig)
