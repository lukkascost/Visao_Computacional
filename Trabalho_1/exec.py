import cv2
import numpy as np


def convolution3x3(img, mask):
    img_copy = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    result = np.zeros(img.shape, dtype=np.uint8)
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
            result[i - 1, j - 1] = np.uint8(sum(p))
    return result


image_original = cv2.imread("01.jpeg", 0)
h = np.ones((3, 3), np.float32) / 9

image = convolution3x3(image_original, h)

cv2.imshow("Filter average my", image)
cv2.imshow("Filter average opencv", cv2.filter2D(image_original, -1, h))
cv2.waitKey(0)
cv2.destroyAllWindows()
