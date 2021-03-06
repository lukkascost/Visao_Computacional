import cv2
import numpy as np


def convolution3x3(image, mask):
    result = np.zeros(image.shape)
    p = np.zeros(9)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            p = np.zeros(9)
            p[0] = image[i - 1, j - 1] * mask[0, 0]
            p[1] = image[i - 1, j] * mask[0, 1]
            p[2] = image[i - 1, j + 1] * mask[0, 2]
            p[3] = image[i, j - 1] * mask[1, 0]
            p[4] = image[i, j] * mask[1, 1]
            p[5] = image[i, j + 1] * mask[1, 2]
            p[6] = image[i + 1, j - 1] * mask[2, 0]
            p[7] = image[i + 1, j] * mask[2, 1]
            p[8] = image[i + 1, j + 1] * mask[2, 2]
            result[i, j] = np.sum(p)
    return result


img = cv2.imread("img/lena.png", 0)

laplace_mask = np.ones((3,3))
laplace_mask[1,1] = -8
laplace = convolution3x3(img, laplace_mask)

cv2.imwrite("img/TRABALHO_04/LENA_ALGORITMO.png", laplace)
cv2.imwrite("img/TRABALHO_04/LENA_NATIVO.png", cv2.filter2D(img, -1, laplace_mask))