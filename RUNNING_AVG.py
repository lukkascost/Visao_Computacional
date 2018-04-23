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

result = np.copy(array)

for i in range(n, array.size):
    result[i] = int(np.average(array[i - n:i]))
result = np.reshape(result, [line, col])

for i in range(1, line, 2):
    result[i] = np.flip(result[i], 0)

print result[0, :10]


result2 = orig.astype(int) - (result.astype(int) * K)
result2[result2 > 0] = 255
result2[result2 <= 0] = 0


cv2.imwrite("res.jpg", result2)



