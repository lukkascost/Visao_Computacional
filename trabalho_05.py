import cv2
import numpy as np


def convolution3x3(img, mask):
    img_copy = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    result = np.zeros(img.shape)
    img_copy[1:-1, 1:-1] = img
    p = np.zeros(9)
    for i in range(1, img_copy.shape[0] - 1):
        for j in range(1, img_copy.shape[1] - 1):
            p[0] = img_copy[i - 1, j - 1] * mask[0, 0]
            p[1] = img_copy[i - 1, j] * mask[0, 1]
            p[2] = img_copy[i - 1, j + 1] * mask[0, 2]
            p[3] = img_copy[i, j - 1] * mask[1, 0]
            p[6] = img_copy[i + 1, j - 1] * mask[2, 0]
            p[4] = img_copy[i, j] * mask[1, 1]
            p[5] = img_copy[i, j + 1] * mask[1, 2]
            p[7] = img_copy[i + 1, j] * mask[2, 1]
            p[8] = img_copy[i + 1, j + 1] * mask[2, 2]
            result[i - 1, j - 1] = sum(p)
    return result


img = cv2.imread("img/01.jpg", 0)

ph_mask = np.matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
pv_mask = np.matrix([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
ph = convolution3x3(img, ph_mask)
pv = convolution3x3(img, pv_mask)
prewit = np.abs(ph) + np.abs(pv)

cv2.imshow("my img", img)
cv2.imshow("ph", ph)
cv2.imshow("pv", pv)
cv2.imshow("prewit", prewit)

cv2.waitKey(0)
cv2.destroyAllWindows()
