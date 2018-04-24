import cv2
import numpy as np

orig = cv2.imread("../DataSet-Baumann/IMAGENS_RECORTADAS_30/c1_p1_00003.JPG", 0)
img = np.copy(orig)
n = 100
K = 0.8
line = img.shape[0]
col = img.shape[1]


for i in range(1, line, 2):
    img[i] = np.flip(img[i], 0)
array = np.reshape(img, img.size)

mapped = np.copy(array)

for i in range(n, array.size):
    mapped[i] = int(np.average(array[i - n:i]))
mapped = np.reshape(mapped, [line, col])

for i in range(1, line, 2):
    mapped[i] = np.flip(mapped[i], 0)

print mapped[0, :10]


result_img = orig.astype(int) - (mapped.astype(int) * K)
result_img[result_img > 0] = 255
result_img[result_img <= 0] = 0


cv2.imwrite("res.jpg", result_img)



